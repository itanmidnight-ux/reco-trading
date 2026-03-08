from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any
import time

import redis
from loguru import logger

from reco_trading.core.microstructure import MicrostructureSnapshot
from reco_trading.execution.execution_firewall import ExecutionFirewall
from reco_trading.execution.idempotent_order_service import IdempotentOrderService
from reco_trading.execution.smart_order_router import SmartOrderRouter, VenueSnapshot
from reco_trading.infra.binance_client import BinanceClient
from reco_trading.infra.database import Database
from reco_trading.infra.exchange_gateway import ExchangeGateway
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
        require_atomic_finalization: bool = False,
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
        self._risk_context: dict[str, float | None] = {'capital_total': 1_000_000.0, 'risk_per_trade': 1.0, 'signal_confidence': 1.0}
        self.last_rejection_reason = ''
        self.last_capital_limited = False
        self.last_allowed_qty = 0.0
        self._symbol_limits_cache: dict[str, float] = {}
        self._idempotent_order_service = IdempotentOrderService(client=self.client, db=self.db, symbol=self.symbol)
        self._gateway = ExchangeGateway(self.client)
        self._execution_lock = asyncio.Lock()
        self._active_execution_context: dict[str, Any] = {}
        self._initialized = False
        self._require_atomic_finalization = bool(require_atomic_finalization)
        try:
            self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
        except Exception:
            self._redis = None

    @property
    def order_service(self) -> IdempotentOrderService:
        return self._idempotent_order_service

    async def get_exchange_min_size(self) -> float:
        if hasattr(self._firewall, 'get_min_size'):
            return await self._firewall.get_min_size(client=self.client, symbol=self.symbol)
        return 0.0

    def update_firewall_limits(
        self,
        *,
        max_total_exposure: float | None = None,
        max_asset_exposure: float | None = None,
        max_daily_loss: float | None = None,
        max_daily_notional: float | None = None,
    ) -> None:
        if hasattr(self._firewall, 'update_limits'):
            self._firewall.update_limits(
                max_total_exposure=max_total_exposure,
                max_asset_exposure=max_asset_exposure,
                max_daily_loss=max_daily_loss,
                max_daily_notional=max_daily_notional,
            )

    async def initialize(self) -> None:
        if not self._redis:
            return
        try:
            await asyncio.to_thread(self._redis.ping)
        except Exception:
            self._redis = None

    def set_risk_context(self, capital_total: float, risk_per_trade: float, signal_confidence: float | None) -> None:
        self._risk_context = {
            'capital_total': max(float(capital_total), 0.0),
            'risk_per_trade': max(float(risk_per_trade), 0.0),
            'signal_confidence': None if signal_confidence is None else float(signal_confidence),
        }

    def _validate_order(self, side: str, amount: float) -> bool:
        normalized_side = str(side).upper()
        if normalized_side not in {'BUY', 'SELL'}:
            self.last_rejection_reason = 'invalid_order_side'
            return False
        if not (0.0 < amount <= self.max_order_size):
            self.last_rejection_reason = 'invalid_order_amount'
            return False
        return True

    @staticmethod
    def _validate_microstructure(microstructure: MicrostructureSnapshot | None) -> bool:
        if microstructure is None:
            return True
        if not isinstance(microstructure, MicrostructureSnapshot):
            return False
        return 0.0 <= microstructure.vpin <= 1.0


    async def get_symbol_min_notional(self, reference_price: float | None = None) -> float:
        cached = self._symbol_limits_cache.get(self.symbol)
        if cached is not None:
            return float(max(cached, 0.0))

        markets = getattr(self.client.exchange, 'markets', None) if hasattr(self.client, 'exchange') else None
        if not markets:
            if hasattr(self.client, 'initialize'):
                await self.client.initialize()
                markets = getattr(self.client.exchange, 'markets', None)

        min_notional = 0.0
        if markets and self.symbol in markets:
            limits = markets[self.symbol].get('limits', {})
            min_cost = limits.get('cost', {}).get('min')
            if min_cost is not None:
                min_notional = float(min_cost)
            elif limits.get('amount', {}).get('min') is not None:
                min_qty = float(limits.get('amount', {}).get('min') or 0.0)
                price = float(reference_price or 0.0)
                if price <= 0.0:
                    try:
                        ticker = await self._gateway.fetch_ticker(self.symbol)
                        price = float(ticker.get('last') or 0.0)
                    except Exception:
                        price = 0.0
                if price > 0.0:
                    min_notional = min_qty * price

        self._symbol_limits_cache[self.symbol] = float(max(min_notional, 0.0))
        return self._symbol_limits_cache[self.symbol]

    async def _evaluate_firewall(self, side: str, amount: float) -> bool:
        decision = await self._firewall.evaluate(client=self.client, symbol=self.symbol, side=side, amount=amount)
        if decision.allowed:
            return True
        self.last_rejection_reason = f'firewall:{decision.reason}'
        self._quant_kernel.on_firewall_rejection(decision.reason, decision.risk_snapshot)
        return False

    async def _apply_capital_limit(self, side: str, amount: float) -> float:
        self.last_capital_limited = False
        self.last_allowed_qty = 0.0
        self._symbol_limits_cache: dict[str, float] = {}

        if side != 'BUY':
            return amount

        confidence = self._risk_context.get('signal_confidence')
        if confidence is None:
            self.last_rejection_reason = 'missing_signal_confidence'
            return 0.0
        confidence = float(confidence)
        if confidence < 0.1 or confidence > 1.0:
            self.last_rejection_reason = 'invalid_signal_confidence'
            return 0.0

        capital_total = float(self._risk_context.get('capital_total') or 0.0)
        risk_per_trade = float(self._risk_context.get('risk_per_trade') or 0.0)
        if capital_total <= 0.0 or risk_per_trade <= 0.0:
            self.last_rejection_reason = 'invalid_risk_context'
            return 0.0

        try:
            ticker = await self._gateway.fetch_ticker(self.symbol)
            last_price = float(ticker.get('last') or 0.0)
        except Exception:
            self.last_rejection_reason = 'ticker_unavailable_for_capital_limit'
            return 0.0

        if last_price <= 0.0:
            self.last_rejection_reason = 'invalid_price_for_capital_limit'
            return 0.0

        max_notional = capital_total * risk_per_trade
        max_qty = max_notional / last_price
        self.last_allowed_qty = max(max_qty, 0.0)
        if max_qty <= 0.0:
            self.last_rejection_reason = 'capital_limit_zero'
            return 0.0
        if amount > max_qty:
            self.last_capital_limited = True
            return max_qty
        return amount

    def _persist_execution(self, payload: dict[str, Any]) -> None:
        if not self._redis:
            return
        try:
            self._redis.set(self._redis_key, json.dumps(payload, ensure_ascii=False))
        except Exception:
            logger.warning('redis write failed')

    async def _persist_execution_async(self, payload: dict[str, Any]) -> None:
        await asyncio.to_thread(self._persist_execution, payload)

    def register_realized_pnl(self, pnl: float) -> None:
        if hasattr(self._firewall, 'register_realized_pnl'):
            self._firewall.register_realized_pnl(float(pnl))


    @staticmethod
    def compute_dynamic_exit_levels(entry_price: float, atr: float, side: str) -> tuple[float, float]:
        atr_val = max(float(atr), 0.0)
        price = max(float(entry_price), 1e-9)
        tp_distance = max(atr_val * 2.4, price * 0.0012)
        sl_distance = max(atr_val * 1.6, price * 0.0010)
        if str(side).upper() == 'BUY':
            return price + tp_distance, price - sl_distance
        return price - tp_distance, price + sl_distance

    async def _execute_child_order(
        self,
        side: str,
        amount: float,
        timeout_seconds: float,
        max_retries: int,
        reference_price: float | None = None,
        decision_tag: str = '',
    ) -> dict[str, Any] | None:
        await self._idempotent_order_service.start()
        decision_timestamp_ms = int(time.time() * 1000)
        decision_bucket = int(decision_timestamp_ms // 1000)
        decision_id = str(self._active_execution_context.get('decision_id') or '')
        decision_context_hash = f'{self.symbol}:{side}:{amount:.8f}:{decision_bucket}:{decision_tag}:{decision_id}'
        client_order_id = self._idempotent_order_service._build_client_order_id(
            side=side,
            amount=amount,
            decision_timestamp_ms=decision_timestamp_ms,
            decision_context_hash=decision_context_hash,
        )
        reservation_notional = 0.0
        for _ in range(max_retries):
            if self._quant_kernel.should_block_trading():
                self.last_rejection_reason = 'blocked_by_quant_kernel'
                return None
            allowed = await self._evaluate_firewall(side, amount)
            if not allowed:
                return None

            try:
                if reference_price is None or reference_price <= 0:
                    ticker = await self._gateway.fetch_ticker(self.symbol)
                    reference_price = float(ticker.get('last') or 0.0)
                sanitized_amount = await self._gateway.sanitize_order_quantity(
                    self.symbol,
                    amount,
                    reference_price=reference_price,
                )
                reservation_notional = max(sanitized_amount * max(float(reference_price or 0.0), 0.0), 0.0)
                if hasattr(self.db, 'reserve_capital'):
                    await self.db.reserve_capital(client_order_id, self.symbol, side, reservation_notional)
                order = await self._idempotent_order_service.submit_market_order(
                    side=side,
                    amount=sanitized_amount,
                    timeout_seconds=timeout_seconds,
                    decision_timestamp_ms=decision_timestamp_ms,
                    client_order_id=client_order_id,
                    decision_context_hash=decision_context_hash,
                    decision_id=decision_id,
                )
            except asyncio.TimeoutError:
                self.last_rejection_reason = 'order_timeout'
                if hasattr(self.db, 'finalize_capital_reservation'):
                    await self.db.finalize_capital_reservation(client_order_id, 0.0, 'released')
                continue
            except Exception:
                self.last_rejection_reason = 'order_rejected_by_exchange_rules'
                if hasattr(self.db, 'finalize_capital_reservation'):
                    await self.db.finalize_capital_reservation(client_order_id, 0.0, 'released')
                return None
            order['decision_id'] = str(self._active_execution_context.get('decision_id') or '')
            await self.db.record_order(order)
            order_id = str(order.get('id', ''))
            if not order_id:
                self.last_rejection_reason = 'empty_order_id'
                return None

            fill = None
            if not hasattr(self.client, 'fetch_order'):
                try:
                    fill = await asyncio.wait_for(self._gateway.wait_for_fill(self.symbol, order_id), timeout=timeout_seconds)
                except asyncio.TimeoutError:
                    fill = None
            else:
                started = time.monotonic()
                while (time.monotonic() - started) < timeout_seconds:
                    current = await self._gateway.fetch_order(self.symbol, order_id)
                    if not current:
                        await asyncio.sleep(1)
                        continue
                    client_id = str(current.get('clientOrderId') or current.get('client_order_id') or client_order_id)
                    if client_id:
                        await self._idempotent_order_service.mark_after_fill(client_id, current)
                    status = str(current.get('status') or '').lower()
                    if status in {'closed', 'filled'}:
                        fill = current
                        break
                    if status in {'canceled', 'cancelled', 'rejected', 'expired'}:
                        fill = current
                        break
                    await asyncio.sleep(1)
            if fill is None:
                self.last_rejection_reason = 'fill_timeout'
                continue
            if not fill:
                self.last_rejection_reason = 'fill_not_received'
                continue
            fill_status = str(fill.get('status') or '').lower()
            if fill_status in {'canceled', 'cancelled', 'rejected', 'expired'}:
                self.last_rejection_reason = f'fill_terminal_{fill_status or "unknown"}'
                if hasattr(self.db, 'finalize_capital_reservation'):
                    await self.db.finalize_capital_reservation(client_order_id, 0.0, 'released')
                continue

            fill['decision_id'] = str(self._active_execution_context.get('decision_id') or '')
            client_order_id = str(order.get('clientOrderId') or order.get('client_order_id') or '')
            if client_order_id:
                await self._idempotent_order_service.mark_after_fill(client_order_id, fill)
            fill_price = float(fill.get('average') or fill.get('price') or 0.0)
            exchange_timestamp_ms = int(fill.get('timestamp') or fill.get('transactTime') or int(time.time() * 1000))
            local_timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            execution_payload = {
                'ts': local_timestamp_ms,
                'symbol': str(fill.get('symbol') or self.symbol),
                'side': str(fill.get('side') or side).upper(),
                'qty': float(fill.get('filled') or fill.get('amount') or sanitized_amount),
                'price': fill_price,
                'status': str(fill.get('status') or 'closed'),
                'pnl': 0.0,
                'decision_id': str(self._active_execution_context.get('decision_id') or ''),
                'exchange_order_id': str(fill.get('id') or order_id),
            }
            if self._require_atomic_finalization and not hasattr(self.db, 'finalize_execution_atomically'):
                self.last_rejection_reason = 'atomic_finalization_unavailable'
                if hasattr(self.db, 'finalize_capital_reservation'):
                    await self.db.finalize_capital_reservation(client_order_id, 0.0, 'released')
                return None

            if hasattr(self.db, 'finalize_execution_atomically'):
                await self.db.finalize_execution_atomically(
                    order=order,
                    fill=fill,
                    execution_payload=execution_payload,
                    client_order_id=client_order_id,
                    committed_notional=max(fill_price, 0.0) * sanitized_amount,
                )
            else:
                await self.db.record_fill(fill)
                await self.db.persist_order_execution(execution_payload)
            logger.info(
                'execution_forensic_log {}',
                json.dumps(
                    {
                        'clientOrderId': client_order_id,
                        'orderId': order_id,
                        'side': str(fill.get('side') or side).upper(),
                        'qty': float(fill.get('filled') or fill.get('amount') or sanitized_amount),
                        'price': fill_price,
                        'expected_edge_gross': float(self._active_execution_context.get('expected_edge') or 0.0),
                        'friction_cost': float(self._active_execution_context.get('friction_cost') or 0.0),
                        'expected_edge_net': float(self._active_execution_context.get('expected_edge_net') or 0.0),
                        'risk_fraction_final': float(self._active_execution_context.get('risk_fraction_final') or self._risk_context.get('risk_per_trade') or 0.0),
                        'portfolio_weight': float(self._active_execution_context.get('portfolio_weight') or 1.0),
                        'drift_score': float(self._active_execution_context.get('drift_score') or 0.0),
                        'timestamp_local_ms': local_timestamp_ms,
                        'timestamp_exchange_ms': exchange_timestamp_ms,
                        'latency_ms': max(local_timestamp_ms - exchange_timestamp_ms, 0),
                    },
                    ensure_ascii=False,
                ),
            )
            self._firewall.register_fill(symbol=self.symbol, notional=max(fill_price, 0.0) * sanitized_amount)
            if self._capital_governor:
                self._capital_governor.register_fill(symbol=self.symbol, exchange='binance', notional=max(fill_price, 0.0) * sanitized_amount)
            if hasattr(self.db, 'finalize_capital_reservation') and not hasattr(self.db, 'finalize_execution_atomically'):
                await self.db.finalize_capital_reservation(client_order_id, max(fill_price, 0.0) * sanitized_amount, 'committed')
            await self._persist_execution_async(
                {
                    'symbol': self.symbol,
                    'side': side,
                    'amount': sanitized_amount,
                    'order_id': order_id,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                }
            )
            self.last_rejection_reason = ''
            return fill
        return None

    async def execute_market_order(
        self,
        side: str,
        amount: float,
        microstructure: MicrostructureSnapshot | None = None,
        timeout_seconds: float | None = None,
        max_retries: int = 5,
        execution_context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        side = str(side).upper()
        if not self._validate_order(side, amount) or not self._validate_microstructure(microstructure):
            return None
        async with self._execution_lock:
            constrained_amount = await self._apply_capital_limit(side=side, amount=amount)
            if constrained_amount <= 0.0:
                return None
            timeout = self.order_timeout_seconds if timeout_seconds is None else max(float(timeout_seconds), 0.1)
            self._active_execution_context = execution_context or {}
            try:
                return await self._execute_child_order(
                    side=side,
                    amount=constrained_amount,
                    timeout_seconds=timeout,
                    max_retries=max_retries,
                    reference_price=None,
                    decision_tag='market',
                )
            finally:
                self._active_execution_context = {}

    async def _execute_institutional_order(self, side: str, amount: float) -> dict[str, Any] | None:
        book = await self._gateway.fetch_order_book(self.symbol, limit=10)
        bids = book.get('bids') or []
        asks = book.get('asks') or []
        bid = float(bids[0][0]) if bids else 0.0
        ask = float(asks[0][0]) if asks else bid
        spread_bps = ((ask - bid) / max(bid, 1e-9)) * 10_000 if bid > 0 else 0.0
        depth = float(sum(float(x[1]) for x in (asks[:5] if side == 'BUY' else bids[:5])))

        ticket = None
        if self._capital_governor:
            self._capital_governor.update_state(
                strategy='directional',
                exchange='binance',
                symbol=self.symbol,
                capital_by_strategy=float(self._risk_context.get('capital_total') or 0.0),
                capital_by_exchange=float(self._risk_context.get('capital_total') or 0.0),
                total_exposure=0.0,
                asset_exposure=0.0,
                exchange_equity=float(self._risk_context.get('capital_total') or 0.0),
            )
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
        for idx, child in enumerate(route):
            fill = await self._execute_child_order(
                side=side,
                amount=float(child['amount']),
                timeout_seconds=self.order_timeout_seconds,
                max_retries=3,
                reference_price=(bid + ask) / 2 if bid > 0 and ask > 0 else None,
                decision_tag=f'institutional:{idx}',
            )
            if fill:
                fills.append(fill)
        if not fills:
            return None
        return {'status': 'institutional_completed', 'fills': fills, 'routed_children': len(route), 'filled_children': len(fills)}

    async def execute(self, side: str, amount: float, execution_context: dict[str, Any] | None = None) -> dict[str, Any] | None:
        side = str(side).upper()
        if not self._initialized:
            await self.initialize()
            self._initialized = True
        if hasattr(self.db, 'execution_advisory_lock'):
            async with self.db.execution_advisory_lock() as lock_acquired:
                if not lock_acquired:
                    self.last_rejection_reason = 'execution_global_lock_not_acquired'
                    return None
                if self._quant_kernel.should_block_trading():
                    self.last_rejection_reason = 'blocked_by_quant_kernel'
                    return None
                if amount >= self.institutional_order_threshold:
                    async with self._execution_lock:
                        return await self._execute_institutional_order(side=side, amount=amount)
                return await self.execute_market_order(side=side, amount=amount, execution_context=execution_context)
        if self._quant_kernel.should_block_trading():
            self.last_rejection_reason = 'blocked_by_quant_kernel'
            return None
        if amount >= self.institutional_order_threshold:
            async with self._execution_lock:
                return await self._execute_institutional_order(side=side, amount=amount)
        return await self.execute_market_order(side=side, amount=amount, execution_context=execution_context)
