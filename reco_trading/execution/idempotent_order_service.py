from __future__ import annotations

import asyncio
import contextlib
import hashlib
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any


class JournalState(str, Enum):
    PENDING_SUBMIT = 'PENDING_SUBMIT'
    ACKED = 'ACKED'
    PARTIAL = 'PARTIAL'
    FILLED = 'FILLED'
    CANCELED = 'CANCELED'
    REJECTED = 'REJECTED'
    UNKNOWN = 'UNKNOWN'


@dataclass(slots=True)
class JournalEntry:
    client_order_id: str
    symbol: str
    side: str
    amount: float
    decision_timestamp_ms: int
    state: JournalState
    exchange_order_id: str = ''


class IdempotentOrderService:
    def __init__(self, client: Any, db: Any, symbol: str, strategy_id: str = 'quant_kernel', reconcile_interval_seconds: float = 10.0) -> None:
        self.client = client
        self.db = db
        self.symbol = symbol
        self.strategy_id = strategy_id
        self._reconcile_interval_seconds = max(float(reconcile_interval_seconds), 1.0)
        self._nonce = 0
        self._lock = asyncio.Lock()
        self._journal: dict[str, JournalEntry] = {}
        self._reconcile_task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        if self._reconcile_task is not None and not self._reconcile_task.done():
            return
        self._stop.clear()
        self._reconcile_task = asyncio.create_task(self._reconcile_loop(), name='idempotent-order-reconcile')

    async def close(self) -> None:
        self._stop.set()
        if self._reconcile_task is not None:
            self._reconcile_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reconcile_task
            self._reconcile_task = None

    def active_trade_in_progress(self) -> bool:
        return any(entry.state in {JournalState.PENDING_SUBMIT, JournalState.ACKED, JournalState.PARTIAL} for entry in self._journal.values())

    def active_client_order_ids(self) -> set[str]:
        return {cid for cid, entry in self._journal.items() if entry.state in {JournalState.PENDING_SUBMIT, JournalState.ACKED, JournalState.PARTIAL}}

    async def submit_market_order(
        self,
        *,
        side: str,
        amount: float,
        timeout_seconds: float,
        decision_timestamp_ms: int,
    ) -> dict[str, Any]:
        async with self._lock:
            client_order_id = self._build_client_order_id(side=side, decision_timestamp_ms=decision_timestamp_ms)
            entry = JournalEntry(
                client_order_id=client_order_id,
                symbol=self.symbol,
                side=side,
                amount=float(amount),
                decision_timestamp_ms=int(decision_timestamp_ms),
                state=JournalState.PENDING_SUBMIT,
            )
            await self._persist_journal(entry)
            self._journal[client_order_id] = entry

        try:
            order = await asyncio.wait_for(self._submit_to_exchange(self.symbol, side, float(amount), client_order_id), timeout=timeout_seconds)
            entry.exchange_order_id = str(order.get('id') or '')
            entry.state = JournalState.ACKED
            await self._persist_journal(entry)
            return order
        except asyncio.TimeoutError:
            recovered = await self._recover_or_resubmit(entry=entry, timeout_seconds=timeout_seconds)
            return recovered
        except Exception:
            entry.state = JournalState.REJECTED
            await self._persist_journal(entry)
            raise

    async def mark_after_fill(self, client_order_id: str, fill: dict[str, Any]) -> None:
        entry = self._journal.get(client_order_id)
        if entry is None:
            return
        status = str(fill.get('status') or '').lower()
        if status in {'closed', 'filled'}:
            entry.state = JournalState.FILLED
        elif status in {'open', 'new'}:
            entry.state = JournalState.PARTIAL if float(fill.get('filled') or 0.0) > 0.0 else JournalState.ACKED
        elif status in {'canceled', 'cancelled'}:
            entry.state = JournalState.CANCELED
        else:
            entry.state = JournalState.UNKNOWN
        await self._persist_journal(entry)

    async def _recover_or_resubmit(self, *, entry: JournalEntry, timeout_seconds: float) -> dict[str, Any]:
        recovered_order = await self.client.fetch_order_by_client_order_id(entry.symbol, entry.client_order_id)
        if recovered_order:
            entry.exchange_order_id = str(recovered_order.get('id') or '')
            entry.state = JournalState.ACKED
            await self._persist_journal(entry)
            return recovered_order
        return await asyncio.wait_for(
            self._submit_to_exchange(entry.symbol, entry.side, entry.amount, entry.client_order_id),
            timeout=timeout_seconds,
        )

    async def _reconcile_loop(self) -> None:
        while not self._stop.is_set():
            try:
                open_orders = await self.client.fetch_open_orders(self.symbol)
                open_client_ids = {str(item.get('clientOrderId') or item.get('client_order_id') or '') for item in open_orders}
                for cid, entry in list(self._journal.items()):
                    if not cid:
                        continue
                    if cid in open_client_ids and entry.state in {JournalState.PENDING_SUBMIT, JournalState.UNKNOWN}:
                        entry.state = JournalState.ACKED
                        await self._persist_journal(entry)
                    if cid not in open_client_ids and entry.state in {JournalState.PENDING_SUBMIT, JournalState.ACKED, JournalState.PARTIAL}:
                        found = await self.client.fetch_order_by_client_order_id(entry.symbol, cid)
                        if found:
                            status = str(found.get('status') or '').lower()
                            if status in {'closed', 'filled'}:
                                entry.state = JournalState.FILLED
                            elif status in {'canceled', 'cancelled'}:
                                entry.state = JournalState.CANCELED
                            else:
                                entry.state = JournalState.UNKNOWN
                            await self._persist_journal(entry)
            except Exception:
                pass
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._reconcile_interval_seconds)
            except asyncio.TimeoutError:
                continue

    async def _persist_journal(self, entry: JournalEntry) -> None:
        if not hasattr(self.db, 'persist_validation_event'):
            return
        await self.db.persist_validation_event(
            {
                'event': 'order_journal',
                'client_order_id': entry.client_order_id,
                'exchange_order_id': entry.exchange_order_id,
                'symbol': entry.symbol,
                'side': entry.side,
                'amount': entry.amount,
                'decision_timestamp_ms': entry.decision_timestamp_ms,
                'state': entry.state.value,
                'ts': int(time.time() * 1000),
            }
        )

    def _build_client_order_id(self, *, side: str, decision_timestamp_ms: int) -> str:
        self._nonce += 1
        decision_bucket = int(decision_timestamp_ms // 1000)
        raw = f'{self.strategy_id}:{self.symbol}:{side}:{decision_bucket}:{self._nonce}'
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:32]

    async def _submit_to_exchange(self, symbol: str, side: str, amount: float, client_order_id: str) -> dict[str, Any]:
        if hasattr(self.client, 'create_market_order_with_client_id'):
            return await self.client.create_market_order_with_client_id(
                symbol,
                side,
                amount,
                client_order_id=client_order_id,
                firewall_checked=True,
            )
        order = await self.client.create_market_order(symbol, side, amount, firewall_checked=True)
        order.setdefault('clientOrderId', client_order_id)
        return order
