from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from loguru import logger


class ExchangeGateway(Protocol):
    async def fetch_balance(self) -> dict[str, dict[str, float]]:
        ...

    async def create_limit_order(self, symbol: str, side: str, amount: float, price: float) -> dict[str, Any]:
        ...

    async def wait_for_fill(self, symbol: str, order_id: str, timeout_seconds: float) -> dict[str, Any] | None:
        ...

    async def cancel_order(self, symbol: str, order_id: str) -> dict[str, Any]:
        ...

    async def create_market_order(self, symbol: str, side: str, amount: float) -> dict[str, Any]:
        ...


@dataclass(slots=True)
class LegPlan:
    exchange: str
    symbol: str
    side: str
    amount: float
    limit_price: float
    reference_price: float
    base_asset: str
    quote_asset: str


@dataclass(slots=True)
class ExecutionConfig:
    sync_window_ms: float = 40.0
    order_timeout_seconds: float = 2.0
    max_slippage_bps: float = 10.0
    imbalance_tolerance: float = 0.000_001
    reduction_factor: float = 0.8
    min_size_multiplier: float = 0.2
    lock_seconds: int = 45


@dataclass(slots=True)
class LegTelemetry:
    submit_ts: float | None = None
    ack_ts: float | None = None
    fill_ts: float | None = None


@dataclass(slots=True)
class LegExecutionResult:
    exchange: str
    symbol: str
    side: str
    requested_amount: float
    filled_amount: float = 0.0
    avg_price: float | None = None
    order_id: str | None = None
    status: str = 'pending'
    error: str | None = None
    cancelled: bool = False
    telemetry: LegTelemetry = field(default_factory=LegTelemetry)


@dataclass(slots=True)
class ArbitrageExecutionReport:
    success: bool
    reason: str
    buy_leg: LegExecutionResult
    sell_leg: LegExecutionResult
    hedge_order: dict[str, Any] | None = None
    pair_locked_until: datetime | None = None


class MultiExchangeArbitrageExecutor:
    def __init__(self, exchanges: dict[str, ExchangeGateway], config: ExecutionConfig | None = None) -> None:
        self.exchanges = exchanges
        self.config = config or ExecutionConfig()
        self._pair_size_multiplier: dict[tuple[str, str], float] = {}
        self._pair_lock_until: dict[tuple[str, str], datetime] = {}
        self.latency_telemetry: list[dict[str, Any]] = []

    async def execute_two_leg_arbitrage(
        self,
        buy_leg: LegPlan,
        sell_leg: LegPlan,
    ) -> ArbitrageExecutionReport:
        pair_key = self._pair_key(buy_leg.exchange, sell_leg.exchange)
        lock_deadline = self._pair_lock_until.get(pair_key)
        if lock_deadline and datetime.now(timezone.utc) < lock_deadline:
            return ArbitrageExecutionReport(
                success=False,
                reason='pair_temporarily_locked',
                buy_leg=self._empty_leg_result(buy_leg, status='blocked'),
                sell_leg=self._empty_leg_result(sell_leg, status='blocked'),
                pair_locked_until=lock_deadline,
            )

        scaled_buy, scaled_sell = self._apply_size_multiplier(pair_key, buy_leg, sell_leg)
        pretrade_ok = await self._validate_pre_trade(scaled_buy, scaled_sell)
        if not pretrade_ok:
            return ArbitrageExecutionReport(
                success=False,
                reason='pretrade_validation_failed',
                buy_leg=self._empty_leg_result(scaled_buy, status='rejected'),
                sell_leg=self._empty_leg_result(scaled_sell, status='rejected'),
            )

        release_event = asyncio.Event()
        buy_task = asyncio.create_task(self._execute_leg(scaled_buy, release_event))
        sell_task = asyncio.create_task(self._execute_leg(scaled_sell, release_event))
        release_event.set()
        buy_result, sell_result = await asyncio.gather(buy_task, sell_task)

        sync_delta_ms = self._sync_delta_ms(buy_result, sell_result)
        if sync_delta_ms > self.config.sync_window_ms:
            logger.warning('Ejecución fuera de ventana de sincronización', delta_ms=sync_delta_ms)
            await self._cancel_if_open(buy_result)
            await self._cancel_if_open(sell_result)
            return ArbitrageExecutionReport(
                success=False,
                reason='sync_window_exceeded',
                buy_leg=buy_result,
                sell_leg=sell_result,
            )

        report = await self._resolve_post_trade(pair_key, buy_result, sell_result)
        self._record_latency_telemetry(report.buy_leg)
        self._record_latency_telemetry(report.sell_leg)
        return report

    async def _validate_pre_trade(self, buy_leg: LegPlan, sell_leg: LegPlan) -> bool:
        if not self._validate_slippage(buy_leg) or not self._validate_slippage(sell_leg):
            return False

        unique_exchanges = {buy_leg.exchange, sell_leg.exchange}
        balances = await asyncio.gather(
            *[self.exchanges[name].fetch_balance() for name in unique_exchanges],
        )
        by_exchange = dict(zip(unique_exchanges, balances))

        buy_balance = by_exchange[buy_leg.exchange]
        sell_balance = by_exchange[sell_leg.exchange]

        quote_free = float(buy_balance.get(buy_leg.quote_asset, {}).get('free', 0.0))
        required_quote = buy_leg.amount * buy_leg.limit_price
        if quote_free < required_quote:
            logger.warning(
                'Balance insuficiente para pata BUY',
                exchange=buy_leg.exchange,
                asset=buy_leg.quote_asset,
                free=quote_free,
                required=required_quote,
            )
            return False

        base_free = float(sell_balance.get(sell_leg.base_asset, {}).get('free', 0.0))
        if base_free < sell_leg.amount:
            logger.warning(
                'Balance insuficiente para pata SELL',
                exchange=sell_leg.exchange,
                asset=sell_leg.base_asset,
                free=base_free,
                required=sell_leg.amount,
            )
            return False
        return True

    def _validate_slippage(self, leg: LegPlan) -> bool:
        if leg.reference_price <= 0:
            return False
        slippage_bps = abs((leg.limit_price - leg.reference_price) / leg.reference_price) * 10_000
        if slippage_bps > self.config.max_slippage_bps:
            logger.warning(
                'Slippage de pata excede máximo permitido',
                exchange=leg.exchange,
                symbol=leg.symbol,
                slippage_bps=slippage_bps,
                max_slippage_bps=self.config.max_slippage_bps,
            )
            return False
        return True

    async def _execute_leg(self, leg: LegPlan, release_event: asyncio.Event) -> LegExecutionResult:
        result = self._empty_leg_result(leg)
        exchange = self.exchanges[leg.exchange]
        await release_event.wait()

        result.telemetry.submit_ts = time.perf_counter()
        try:
            order = await asyncio.wait_for(
                exchange.create_limit_order(leg.symbol, leg.side, leg.amount, leg.limit_price),
                timeout=self.config.order_timeout_seconds,
            )
        except asyncio.TimeoutError:
            result.status = 'timeout_on_submit'
            result.error = 'submit_timeout'
            return result
        except Exception as exc:
            result.status = 'submit_error'
            result.error = str(exc)
            return result

        result.telemetry.ack_ts = time.perf_counter()
        result.order_id = str(order.get('id', '')) or None
        result.status = str(order.get('status', 'submitted'))

        if not result.order_id:
            result.status = 'invalid_order'
            result.error = 'missing_order_id'
            return result

        try:
            fill = await asyncio.wait_for(
                exchange.wait_for_fill(leg.symbol, result.order_id, self.config.order_timeout_seconds),
                timeout=self.config.order_timeout_seconds,
            )
        except asyncio.TimeoutError:
            result.status = 'timeout_on_fill'
            result.error = 'fill_timeout'
            return result
        except Exception as exc:
            result.status = 'fill_error'
            result.error = str(exc)
            return result

        if not fill:
            result.status = 'not_filled'
            return result

        result.telemetry.fill_ts = time.perf_counter()
        result.filled_amount = float(fill.get('filled', 0.0) or 0.0)
        result.avg_price = float(fill.get('average', leg.limit_price) or leg.limit_price)
        result.status = str(fill.get('status', 'filled'))
        return result

    async def _resolve_post_trade(
        self,
        pair_key: tuple[str, str],
        buy_result: LegExecutionResult,
        sell_result: LegExecutionResult,
    ) -> ArbitrageExecutionReport:
        buy_filled = buy_result.filled_amount
        sell_filled = sell_result.filled_amount
        imbalance = abs(buy_filled - sell_filled)

        both_filled = buy_filled > 0 and sell_filled > 0
        if both_filled and imbalance <= self.config.imbalance_tolerance:
            return ArbitrageExecutionReport(
                success=True,
                reason='filled',
                buy_leg=buy_result,
                sell_leg=sell_result,
            )

        await self._cancel_if_open(buy_result)
        await self._cancel_if_open(sell_result)

        hedge_order = await self._run_emergency_hedge(buy_result, sell_result)
        lock_until = self._activate_contingency(pair_key)
        return ArbitrageExecutionReport(
            success=False,
            reason='fill_imbalance_contingency',
            buy_leg=buy_result,
            sell_leg=sell_result,
            hedge_order=hedge_order,
            pair_locked_until=lock_until,
        )

    async def _cancel_if_open(self, result: LegExecutionResult) -> None:
        if not result.order_id or result.status in {'filled', 'closed', 'canceled'}:
            return
        exchange = self.exchanges[result.exchange]
        try:
            await exchange.cancel_order(result.symbol, result.order_id)
            result.cancelled = True
            result.status = 'canceled'
        except Exception as exc:
            logger.warning('No se pudo cancelar orden abierta', order_id=result.order_id, error=str(exc))

    async def _run_emergency_hedge(
        self,
        buy_result: LegExecutionResult,
        sell_result: LegExecutionResult,
    ) -> dict[str, Any] | None:
        imbalance = buy_result.filled_amount - sell_result.filled_amount
        if abs(imbalance) <= self.config.imbalance_tolerance:
            return None

        if imbalance > 0:
            hedge_exchange_name = buy_result.exchange
            hedge_side = 'SELL'
            hedge_amount = imbalance
            hedge_symbol = buy_result.symbol
        else:
            hedge_exchange_name = sell_result.exchange
            hedge_side = 'BUY'
            hedge_amount = abs(imbalance)
            hedge_symbol = sell_result.symbol

        try:
            hedge_order = await self.exchanges[hedge_exchange_name].create_market_order(
                hedge_symbol,
                hedge_side,
                hedge_amount,
            )
            logger.warning(
                'Hedge de emergencia ejecutado',
                exchange=hedge_exchange_name,
                side=hedge_side,
                amount=hedge_amount,
            )
            return hedge_order
        except Exception as exc:
            logger.exception('Fallo en hedge de emergencia', error=str(exc))
            return None

    def _activate_contingency(self, pair_key: tuple[str, str]) -> datetime:
        current_multiplier = self._pair_size_multiplier.get(pair_key, 1.0)
        reduced_multiplier = max(self.config.min_size_multiplier, current_multiplier * self.config.reduction_factor)
        self._pair_size_multiplier[pair_key] = reduced_multiplier

        lock_until = datetime.now(timezone.utc) + timedelta(seconds=self.config.lock_seconds)
        self._pair_lock_until[pair_key] = lock_until
        return lock_until

    def _apply_size_multiplier(self, pair_key: tuple[str, str], buy_leg: LegPlan, sell_leg: LegPlan) -> tuple[LegPlan, LegPlan]:
        multiplier = self._pair_size_multiplier.get(pair_key, 1.0)
        if multiplier >= 0.999:
            return buy_leg, sell_leg

        return (
            replace(buy_leg, amount=buy_leg.amount * multiplier),
            replace(sell_leg, amount=sell_leg.amount * multiplier),
        )

    def _pair_key(self, first: str, second: str) -> tuple[str, str]:
        return tuple(sorted((first, second)))

    def _sync_delta_ms(self, left: LegExecutionResult, right: LegExecutionResult) -> float:
        left_ts = left.telemetry.submit_ts
        right_ts = right.telemetry.submit_ts
        if left_ts is None or right_ts is None:
            return float('inf')
        return abs(left_ts - right_ts) * 1000.0

    def _record_latency_telemetry(self, leg: LegExecutionResult) -> None:
        self.latency_telemetry.append(
            {
                'exchange': leg.exchange,
                'symbol': leg.symbol,
                'side': leg.side,
                'submit_ts': leg.telemetry.submit_ts,
                'ack_ts': leg.telemetry.ack_ts,
                'fill_ts': leg.telemetry.fill_ts,
            }
        )

    def _empty_leg_result(self, leg: LegPlan, status: str = 'pending') -> LegExecutionResult:
        return LegExecutionResult(
            exchange=leg.exchange,
            symbol=leg.symbol,
            side=leg.side,
            requested_amount=leg.amount,
            status=status,
        )
