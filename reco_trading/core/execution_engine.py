from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

import redis
from loguru import logger

from reco_trading.core.microstructure import MicrostructureSnapshot
from reco_trading.execution.execution_firewall import ExecutionFirewall
from reco_trading.execution.smart_order_router import SmartOrderRouter, VenueSnapshot
from reco_trading.infra.binance_client import BinanceClient
from reco_trading.infra.database import Database
from reco_trading.kernel.capital_governor import CapitalGovernor


class _NullKernel:
    def should_block_trading(self) -> bool:
        return False

    def on_firewall_rejection(self, reason: str, risk_snapshot: dict[str, Any]) -> None:
        logger.warning('Firewall rejection', reason=reason, risk_snapshot=risk_snapshot)


class ExecutionEngine:
    def __init__(
        self,
        client: BinanceClient,
        symbol: str,
        db: Database,
        redis_url: str = 'redis://localhost:6379/0',
        redis_key: str = 'reco_trading:last_execution',
        max_order_size: float = 100_000.0,
        institutional_order_threshold: float = 25.0,
        order_timeout_seconds: float = 30.0,
        firewall: ExecutionFirewall | None = None,
        quant_kernel: Any | None = None,
        sor: SmartOrderRouter | None = None,
        capital_governor: CapitalGovernor | None = None,
    ) -> None:
        self.client = client
        self.symbol = symbol
        self.db = db
        self.max_order_size = max_order_size
        self.institutional_order_threshold = institutional_order_threshold
        self.order_timeout_seconds = max(float(order_timeout_seconds), 0.1)
        self._firewall = firewall or ExecutionFirewall()
        self._quant_kernel = quant_kernel or _NullKernel()
        self._capital_governor = capital_governor
        self._sor = sor or SmartOrderRouter(capital_governor=capital_governor)
        self._redis_key = redis_key
        try:
            self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
            self._redis.ping()
        except Exception:
            self._redis = None

    def _validate_order(self, side: str, amount: float) -> bool:
        normalized_side = str(side).upper()
        if normalized_side not in {'BUY', 'SELL'}:
            return False
        return 0.0 < amount <= self.max_order_size

    @staticmethod
    def _validate_microstructure(microstructure: MicrostructureSnapshot | None) -> bool:
        if microstructure is None:
            return True
        if not isinstance(microstructure, MicrostructureSnapshot):
            return False
        return 0.0 <= microstructure.vpin <= 1.0

    async def _evaluate_firewall(self, side: str, amount: float) -> bool:
        decision = await self._firewall.evaluate(client=self.client, symbol=self.symbol, side=side, amount=amount)
        if decision.allowed:
            return True
        self._quant_kernel.on_firewall_rejection(decision.reason, decision.risk_snapshot)
        return False

    def _persist_execution(self, payload: dict[str, Any]) -> None:
        if not self._redis:
            return
        try:
            self._redis.set(self._redis_key, json.dumps(payload, ensure_ascii=False))
        except Exception:
            logger.warning('redis write failed')

    async def _execute_child_order(
        self,
        side: str,
        amount: float,
        timeout_seconds: float,
        max_retries: int,
        reference_price: float | None = None,
    ) -> dict[str, Any] | None:
        for _ in range(max_retries):
            if self._quant_kernel.should_block_trading():
                return None
            try:
                allowed = await asyncio.wait_for(self._evaluate_firewall(side, amount), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                continue
            if not allowed:
                return None

            try:
                if reference_price is None or reference_price <= 0:
                    ticker = await self.client.fetch_ticker(self.symbol)
                    reference_price = float(ticker.get('last') or 0.0)
                sanitized_amount = await self.client.sanitize_order_quantity(
                    self.symbol,
                    amount,
                    reference_price=reference_price,
                )
                order = await asyncio.wait_for(
                    self.client.create_market_order(self.symbol, side, sanitized_amount, firewall_checked=True),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                continue
            await self.db.record_order(order)
            order_id = str(order.get('id', ''))
            if not order_id:
                return None

            try:
                fill = await asyncio.wait_for(self.client.wait_for_fill(self.symbol, order_id), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                continue
            if not fill:
                continue

            await self.db.record_fill(fill)
            fill_price = float(fill.get('average') or fill.get('price') or 0.0)
            await self.db.persist_order_execution(
                {
                    'ts': int(datetime.now(timezone.utc).timestamp() * 1000),
                    'symbol': str(fill.get('symbol') or self.symbol),
                    'side': str(fill.get('side') or side).upper(),
                    'qty': float(fill.get('filled') or fill.get('amount') or sanitized_amount),
                    'price': fill_price,
                    'status': str(fill.get('status') or 'closed'),
                    'pnl': 0.0,
                }
            )
            self._firewall.register_fill(symbol=self.symbol, notional=max(fill_price, 0.0) * sanitized_amount)
            if self._capital_governor:
                self._capital_governor.register_fill(symbol=self.symbol, exchange='binance', notional=max(fill_price, 0.0) * sanitized_amount)
            self._persist_execution(
                {
                    'symbol': self.symbol,
                    'side': side,
                    'amount': sanitized_amount,
                    'order_id': order_id,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                }
            )
            return fill
        return None

    async def execute_market_order(
        self,
        side: str,
        amount: float,
        microstructure: MicrostructureSnapshot | None = None,
        timeout_seconds: float | None = None,
        max_retries: int = 5,
    ) -> dict[str, Any] | None:
        side = str(side).upper()
        if not self._validate_order(side, amount) or not self._validate_microstructure(microstructure):
            return None
        timeout = self.order_timeout_seconds if timeout_seconds is None else max(float(timeout_seconds), 0.1)
        return await self._execute_child_order(
            side=side,
            amount=amount,
            timeout_seconds=timeout,
            max_retries=max_retries,
            reference_price=None,
        )

    async def _execute_institutional_order(self, side: str, amount: float) -> dict[str, Any] | None:
        book = await self.client.fetch_order_book(self.symbol, limit=10)
        bids = book.get('bids') or []
        asks = book.get('asks') or []
        bid = float(bids[0][0]) if bids else 0.0
        ask = float(asks[0][0]) if asks else bid
        spread_bps = ((ask - bid) / max(bid, 1e-9)) * 10_000 if bid > 0 else 0.0
        depth = float(sum(float(x[1]) for x in (asks[:5] if side == 'BUY' else bids[:5])))

        ticket = None
        if self._capital_governor:
            ticket = self._capital_governor.issue_ticket(
                strategy='directional',
                exchange='binance',
                symbol=self.symbol,
                requested_notional=amount * max((bid + ask) / 2, 1.0),
                pnl_or_returns=[],
                spread_bps=spread_bps,
                available_liquidity=depth,
                price_gap_pct=0.01,
            )

        route = self._sor.route_order(
            amount=amount,
            venues=[
                VenueSnapshot('binance_spot', spread_bps=spread_bps, depth=max(depth, 1e-9), latency_ms=30.0, fee_bps=10.0, fill_ratio=0.98, liquidity=max(depth, amount)),
            ],
            strategy='VWAP',
            slices=3,
            capital_ticket=ticket,
        )
        fills: list[dict[str, Any]] = []
        for child in route:
            fill = await self._execute_child_order(
                side=side,
                amount=float(child['amount']),
                timeout_seconds=self.order_timeout_seconds,
                max_retries=3,
                reference_price=(bid + ask) / 2 if bid > 0 and ask > 0 else None,
            )
            if fill:
                fills.append(fill)
        if not fills:
            return None
        return {'status': 'institutional_completed', 'fills': fills, 'routed_children': len(route), 'filled_children': len(fills)}

    async def execute(self, side: str, amount: float) -> dict[str, Any] | None:
        side = str(side).upper()
        if self._quant_kernel.should_block_trading():
            return None
        if amount >= self.institutional_order_threshold:
            return await self._execute_institutional_order(side=side, amount=amount)
        return await self.execute_market_order(side=side, amount=amount)
