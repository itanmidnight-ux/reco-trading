from __future__ import annotations

import asyncio
import signal
import time
import traceback
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from reco_trading.config.settings import get_settings
from reco_trading.core.data_buffer import DataBuffer
from reco_trading.core.execution_engine import ExecutionEngine
from reco_trading.core.feature_engine import FeatureEngine
from reco_trading.core.market_data import MarketDataService, MarketQuality
from reco_trading.core.market_regime import MarketRegimeDetector
from reco_trading.core.mean_reversion_model import MeanReversionModel
from reco_trading.core.momentum_model import MomentumModel
from reco_trading.core.signal_fusion import SignalCombiner
from reco_trading.core.system_state import SystemState
from reco_trading.execution.execution_firewall import ExecutionFirewall
from reco_trading.infra.binance_client import BinanceClient
from reco_trading.infra.database import Database
from reco_trading.monitoring.metrics import MetricsExporter, TradingMetrics
from reco_trading.ui.terminal_dashboard import TerminalDashboard
from reco_trading.ui.visual_snapshot import VisualSnapshot


class PositionState(Enum):
    FLAT = 'flat'
    LONG = 'long'


@dataclass(slots=True)
class RuntimeState:
    equity: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    fees_paid: float = 0.0
    capital_in_position: float = 0.0
    trades: int = 0
    winning_trades: int = 0
    consecutive_losses: int = 0
    consecutive_cycle_errors: int = 0
    last_block_reason: str = 'booting'
    rejection_count: int = 0
    kill_switch: bool = False
    avg_latency_ms: float = 0.0

    position_state: PositionState = PositionState.FLAT
    position_qty: float = 0.0
    entry_price: float = 0.0
    position_opened_at: datetime | None = None
    tp_price: float = 0.0
    sl_price: float = 0.0


class SignalEngine:
    def __init__(self) -> None:
        self.feature_engine = FeatureEngine()
        self.momentum = MomentumModel()
        self.reversion = MeanReversionModel()
        self._normal = NormalDist(mu=0.0, sigma=1.0)

    def generate(self, ohlcv: pd.DataFrame, spread_bps: float) -> dict[str, Any]:
        feats = self.feature_engine.build(ohlcv)
        if feats.empty:
            raise ValueError('Features vacías')

        snapshot = self.feature_engine.market_snapshot(ohlcv, spread=spread_bps)
        model_momentum = float(np.clip(self.momentum.predict_from_snapshot(snapshot), 0.0, 1.0))
        model_reversion = float(np.clip(self.reversion.predict_from_snapshot(snapshot), 0.0, 1.0))

        returns = pd.Series(snapshot.returns)
        n = max(int(returns.size), 1)
        mu = float(returns.mean() or 0.0)
        sigma = float(returns.std() or 1e-9)
        rolling_vol = float(returns.tail(40).std() or sigma)
        skew = float(returns.skew() or 0.0)
        kurtosis = float(returns.kurtosis() or 0.0)

        # estadística real: t-score del drift de retornos log
        edge_z = float(np.clip((mu / max(sigma / np.sqrt(n), 1e-9)), -8.0, 8.0))
        p_drift_up = float(self._normal.cdf(edge_z))

        # mean reversion estadístico: distancia a VWAP normalizada por volatilidad
        reversion_z = float(np.clip((-snapshot.vwap_distance) / max(rolling_vol, 1e-9), -8.0, 8.0))
        p_reversion_up = float(self._normal.cdf(reversion_z))

        # combinación entre señal estadística y modelo entrenado
        momentum_prob = float(np.clip(0.65 * p_drift_up + 0.35 * model_momentum, 0.0, 1.0))
        reversion_prob = float(np.clip(0.65 * p_reversion_up + 0.35 * model_reversion, 0.0, 1.0))

        return {
            'snapshot': snapshot,
            'model_scores': {'momentum': momentum_prob, 'mean_reversion': reversion_prob},
            'atr': snapshot.atr,
            'returns': snapshot.returns,
            'prices': feats['close'].tail(300),
            'mu': mu,
            'sigma': sigma,
            'volatility': rolling_vol,
            'skew': skew,
            'kurtosis': kurtosis,
            'edge_z': edge_z,
        }


class DecisionEngine:
    def __init__(self, min_edge: float = 0.08) -> None:
        self.min_edge = float(min_edge)

    def decide(
        self,
        *,
        probability: float,
        expected_edge: float,
        friction_cost: float,
        trading_enabled: bool,
        market_operable: bool,
        force_hold: bool = False,
        reason_prefix: str = '',
    ) -> tuple[str, float, str]:
        p = float(np.clip(probability, 0.0, 1.0))
        if not trading_enabled:
            return 'DISABLE_TRADING', p, f'{reason_prefix}kill_switch_active'
        if force_hold:
            return 'HOLD', p, f'{reason_prefix}learning_market'
        if not market_operable:
            return 'HOLD', p, f'{reason_prefix}market_not_operable'
        if expected_edge <= friction_cost:
            return 'HOLD', p, f'{reason_prefix}edge_below_friction edge={expected_edge:.6f} friction={friction_cost:.6f}'
        if abs(expected_edge) < self.min_edge:
            return 'HOLD', p, f'{reason_prefix}insufficient_edge edge={expected_edge:+.6f}'
        if expected_edge > 0:
            return 'BUY', p, f'{reason_prefix}positive_edge edge={expected_edge:+.6f}'
        return 'SELL', p, f'{reason_prefix}negative_edge edge={expected_edge:+.6f}'


class QuantKernel:
    MIN_SECONDS_BETWEEN_TRADES = 30
    MAX_CONSECUTIVE_CYCLE_ERRORS = 5

    def __init__(self) -> None:
        self.s = get_settings()
        self.state = RuntimeState()
        self.last_trade_ts: float | None = None
        self.initial_equity = 0.0
        self._shutdown_reason = 'running'
        self._last_market_quality = MarketQuality(True, 'booting', 0.0, 0.0, 0.0, 0.0)

        self.metrics = TradingMetrics()
        self.metrics_exporter = MetricsExporter(port=self.s.monitoring_metrics_port, addr=self.s.monitoring_metrics_host)
        self.dashboard = TerminalDashboard()
        self.shutdown_event = asyncio.Event()
        self.system_state = SystemState.WAITING_FOR_DATA.value
        self.activity_text = 'Inicializando kernel'
        self.learning_started_at_ms: int | None = None
        self.execution_status = 'IDLE'

    async def initialize(self) -> None:
        self.client = BinanceClient(
            self.s.binance_api_key.get_secret_value(),
            self.s.binance_api_secret.get_secret_value(),
            testnet=self.s.binance_testnet,
            confirm_mainnet=self.s.confirm_mainnet,
        )
        self.db = Database(self.s.postgres_dsn, self.s.postgres_admin_dsn)
        self.market_data = MarketDataService(self.client, self.s.symbol, self.s.timeframe)
        self.signal_engine = SignalEngine()
        self.data_buffer = DataBuffer(window_seconds=self.s.learning_phase_seconds)
        self.signal_combiner = SignalCombiner()
        self.decision_engine = DecisionEngine()
        self.regime_detector = MarketRegimeDetector(n_states=3)

        self.execution_engine = ExecutionEngine(
            self.client,
            self.s.symbol,
            self.db,
            redis_url=self.s.redis_url,
            order_timeout_seconds=self.s.execution_order_timeout_seconds,
            firewall=ExecutionFirewall(
                max_total_exposure=100_000,
                max_asset_exposure=50_000,
                max_daily_loss=5_000,
                max_daily_notional=200_000,
            ),
            quant_kernel=self,
        )

        await self.client.ping()
        ok, free_usdt, reason = await self._fetch_account_balance()
        if not ok:
            raise RuntimeError(f'Balance inicial inválido: {reason}')
        self.state.equity = free_usdt
        self.initial_equity = free_usdt
        self.state.last_block_reason = 'none'
        logger.info(f'Balance REAL sincronizado: {free_usdt:.2f} USDT')

    async def _fetch_account_balance(self) -> tuple[bool, float, str]:
        try:
            balance = await self.client.fetch_balance()
        except Exception as exc:
            return False, 0.0, f'binance_balance_error:{exc}'
        usdt = float((balance.get('USDT') or {}).get('free') or 0.0)
        if not np.isfinite(usdt) or usdt <= 0.0:
            return False, 0.0, 'invalid_usdt_balance'
        return True, usdt, 'ok'

    def _map_regime(self, raw_regime: str) -> str:
        r = str(raw_regime).lower()
        if r == 'trend':
            return 'TREND'
        if r in {'range', 'low_volatility'}:
            return 'RANGE'
        if r in {'high_volatility', 'volatile'}:
            return 'HIGH_VOL'
        return 'RANGE'

    def _should_activate_kill_switch(self) -> tuple[bool, str]:
        if self.shutdown_event.is_set():
            return True, 'shutdown_requested'
        if self.state.kill_switch:
            return True, 'kill_switch_active'
        if self.state.rejection_count >= self.s.kill_switch_max_rejections:
            self.state.kill_switch = True
            return True, 'too_many_rejections'
        if self.state.avg_latency_ms >= self.s.kill_switch_max_latency_ms:
            self.state.kill_switch = True
            return True, 'latency_circuit_breaker'
        if self.state.consecutive_losses >= self.s.max_consecutive_losses:
            self.state.kill_switch = True
            return True, 'too_many_consecutive_losses'
        if self.initial_equity > 0:
            total_equity = self.state.equity + self.state.unrealized_pnl + self.state.realized_pnl - self.state.fees_paid
            if not np.isfinite(total_equity):
                self.state.kill_switch = True
                return True, 'equity_inconsistent'
            drawdown = 1.0 - (total_equity / max(self.initial_equity, 1e-9))
            if drawdown >= self.s.max_global_drawdown:
                self.state.kill_switch = True
                return True, 'drawdown_circuit_breaker'
        return False, 'none'

    def should_block_trading(self) -> bool:
        blocked, reason = self._should_activate_kill_switch()
        if blocked:
            self.state.last_block_reason = reason
            self.system_state = 'KILL_SWITCH_ACTIVE'
        return blocked

    def on_firewall_rejection(self, reason: str, risk_snapshot: dict[str, Any]) -> None:
        self.state.rejection_count += 1
        self.state.last_block_reason = f'firewall:{reason}'
        self.system_state = SystemState.BLOCKED_BY_RISK.value
        logger.warning('Execution blocked by firewall', reason=reason, risk_snapshot=risk_snapshot)

    async def _execute_order(self, side: str, qty: float) -> dict[str, Any] | None:
        if qty <= 0.0:
            self.state.last_block_reason = 'invalid_qty'
            return None

        self.execution_status = 'ORDER_SENT'
        started = time.perf_counter()
        fill = await self.execution_engine.execute(side, qty)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.state.avg_latency_ms = (self.state.avg_latency_ms * 0.8) + (elapsed_ms * 0.2)

        if not fill:
            self.state.rejection_count += 1
            self.execution_status = 'ORDER_REJECTED'
            self.state.last_block_reason = self.execution_engine.last_rejection_reason or 'execution_returned_none'
            return None

        self.execution_status = 'ORDER_FILLED'
        if fill.get('status') == 'institutional_completed':
            fills = fill.get('fills') or []
            if not fills:
                self.state.last_block_reason = 'institutional_no_fills'
                return None
            fill = fills[-1]

        executed_qty = float(fill.get('filled') or fill.get('amount') or 0.0)
        if executed_qty <= 0:
            self.state.last_block_reason = 'execution_unfilled'
            return None

        return {
            'id': str(fill.get('id') or ''),
            'symbol': str(fill.get('symbol') or self.s.symbol),
            'side': str(fill.get('side') or side).upper(),
            'qty': executed_qty,
            'price': float(fill.get('average') or fill.get('price') or 0.0),
            'status': str(fill.get('status') or 'closed').lower(),
            'pnl': 0.0,
        }

    def _risk_fraction(self, expected_edge: float, volatility: float) -> float:
        base = float(np.clip(self.s.risk_per_trade, 0.001, self.s.max_confidence_allocation))
        edge_factor = float(np.clip(abs(expected_edge) / 0.20, 0.10, 2.0))
        vol_factor = float(np.clip(self.s.volatility_target / max(volatility, 1e-6), 0.20, 1.5))
        return float(np.clip(base * edge_factor * vol_factor, 0.0, self.s.max_confidence_allocation))

    def _publish_dashboard(
        self,
        decision: str,
        confidence: float,
        expected_edge: float,
        mom: float,
        rev: float,
        reg_prob: float,
        regime: str,
        last_price: float,
        binance_state: str,
        learning_remaining_seconds: float = 0.0,
    ) -> None:
        total_equity = self.state.equity + self.state.realized_pnl + self.state.unrealized_pnl - self.state.fees_paid
        drawdown = 0.0 if self.initial_equity <= 0 else max(0.0, 1.0 - (total_equity / max(self.initial_equity, 1e-9)))
        now = datetime.now(timezone.utc)
        position_time = (now - self.state.position_opened_at).total_seconds() if self.state.position_opened_at else 0.0
        cooldown = 0.0
        if self.last_trade_ts:
            cooldown = max(self.MIN_SECONDS_BETWEEN_TRADES - (now.timestamp() - self.last_trade_ts), 0.0)
        self.dashboard.update(
            VisualSnapshot(
                capital=total_equity,
                balance=self.state.equity,
                pnl_total=self.state.realized_pnl + self.state.unrealized_pnl - self.state.fees_paid,
                pnl_diario=self.state.realized_pnl,
                drawdown=drawdown,
                riesgo_activo=float(np.clip(self.s.risk_per_trade, 0.0, 1.0)),
                exposicion=self.state.capital_in_position,
                trades=self.state.trades,
                win_rate=(self.state.winning_trades / self.state.trades if self.state.trades else 0.0),
                expectancy=expected_edge,
                sharpe_rolling=0.0,
                regimen=regime,
                senal=decision,
                latencia_ms=self.state.avg_latency_ms,
                ultimo_precio=max(last_price, 0.0),
                estado_binance=binance_state,
                estado_sistema=self.system_state,
                actividad=self.activity_text,
                motivo_bloqueo=self.state.last_block_reason,
                confianza=confidence,
                tiempo_en_posicion_s=position_time,
                cooldown_restante_s=cooldown,
                score_momentum=mom,
                score_reversion=rev,
                score_regime=reg_prob,
                learning_remaining_seconds=learning_remaining_seconds,
                spread_bps=self._last_market_quality.spread_bps,
                slippage_bps=float(self.s.slippage_bps),
                estimated_fees=self.state.fees_paid,
                execution_status=self.execution_status,
            )
        )

    def _handle_cycle_exception(self, exc: Exception) -> bool:
        self.state.consecutive_cycle_errors += 1
        self.activity_text = f'Error de ciclo: {exc}'
        self.execution_status = 'ERROR'
        self.system_state = SystemState.ERROR.value
        self._publish_dashboard('HOLD', 0.0, 0.0, 0.5, 0.5, 0.5, 'ERROR', 0.0, 'ERROR')
        if self.state.consecutive_cycle_errors >= self.MAX_CONSECUTIVE_CYCLE_ERRORS:
            self.state.kill_switch = True
            self._shutdown_reason = 'max_consecutive_cycle_errors'
            self.shutdown_event.set()
            return True
        return False

    async def run(self) -> None:
        await self.initialize()
        await self.db.init()
        self.metrics_exporter.start()
        self.dashboard.start()
        self._install_signal_handlers()

        while not self.shutdown_event.is_set():
            try:
                self.system_state = SystemState.WAITING_FOR_DATA.value
                self.execution_status = 'IDLE'
                self.activity_text = 'Esperando OHLCV desde Binance'
                now = datetime.now(timezone.utc)

                ohlcv = await self.market_data.latest_ohlcv(limit=300)
                self.data_buffer.push_ohlcv(ohlcv)
                if self.learning_started_at_ms is None:
                    self.learning_started_at_ms = int(ohlcv['timestamp'].iloc[0].timestamp() * 1000)

                try:
                    spread_bps = await self.market_data.latest_spread_bps()
                except Exception:
                    spread_bps = 0.0
                self.data_buffer.record_spread(spread_bps)

                self._last_market_quality = self.market_data.assess_market_quality(
                    self.data_buffer.ohlcv,
                    spread_bps=spread_bps,
                    max_spread_bps=self.s.market_max_spread_bps,
                    max_volatility=self.s.market_max_realized_volatility,
                    min_avg_volume=self.s.market_min_avg_volume,
                    max_gap_ratio=self.s.market_max_gap_ratio,
                )

                progress, remaining = self.data_buffer.learning_progress(self.learning_started_at_ms, now.timestamp())
                if remaining > 0.0:
                    self.system_state = SystemState.LEARNING_MARKET.value
                    decision, confidence, reason = self.decision_engine.decide(
                        probability=0.5,
                        expected_edge=0.0,
                        friction_cost=0.0,
                        trading_enabled=not self.state.kill_switch,
                        market_operable=False,
                        force_hold=True,
                        reason_prefix='phase=learning;',
                    )
                    self.state.last_block_reason = reason
                    self.activity_text = f'APRENDIENDO MERCADO ({progress:.0%}) restante={remaining:.1f}s'
                    self._publish_dashboard(decision, confidence, 0.0, 0.5, 0.5, 0.5, 'RANGE', float(ohlcv["close"].iloc[-1]), 'OK', remaining)
                    await asyncio.sleep(self.s.loop_interval_seconds)
                    continue

                if self.should_block_trading():
                    self._publish_dashboard('DISABLE_TRADING', 0.0, 0.0, 0.5, 0.5, 0.5, 'RANGE', float(ohlcv["close"].iloc[-1]), 'OK')
                    await asyncio.sleep(self.s.loop_interval_seconds)
                    continue

                self.system_state = SystemState.ANALYZING_MARKET.value
                if not self._last_market_quality.operable:
                    self.system_state = SystemState.WAITING_EDGE.value
                    self.state.last_block_reason = self._last_market_quality.reason
                    self.activity_text = f'Mercado no operable: {self._last_market_quality.reason}'
                    self._publish_dashboard('HOLD', 0.0, 0.0, 0.5, 0.5, 0.5, 'RANGE', float(ohlcv["close"].iloc[-1]), 'OK')
                    await asyncio.sleep(self.s.loop_interval_seconds)
                    continue

                ok_balance, free_usdt, reason = await self._fetch_account_balance()
                if not ok_balance:
                    self.system_state = SystemState.BLOCKED_BY_RISK.value
                    self.state.last_block_reason = reason
                    self.activity_text = f'Balance inválido: {reason}'
                    self._publish_dashboard('DISABLE_TRADING', 0.0, 0.0, 0.5, 0.5, 0.5, 'RANGE', float(ohlcv["close"].iloc[-1]), 'ERROR')
                    await asyncio.sleep(self.s.loop_interval_seconds)
                    continue

                self.state.equity = free_usdt
                sig = self.signal_engine.generate(self.data_buffer.ohlcv, spread_bps=spread_bps)
                last_price = float(sig['prices'].iloc[-1])

                if self.state.position_state == PositionState.LONG:
                    self.state.unrealized_pnl = (last_price - self.state.entry_price) * self.state.position_qty
                    self.state.capital_in_position = self.state.position_qty * last_price
                else:
                    self.state.unrealized_pnl = 0.0
                    self.state.capital_in_position = 0.0

                regime_raw = self.regime_detector.predict(sig['returns'], sig['prices']).get('regime', 'range')
                regime = self._map_regime(regime_raw)
                regime_prob = 0.78 if regime == 'TREND' else (0.62 if regime == 'RANGE' else 0.55)
                breakdown = self.signal_combiner.combine(
                    sig['model_scores']['momentum'],
                    sig['model_scores']['mean_reversion'],
                    regime_prob,
                    regime,
                )

                probability = float(breakdown.combined)
                expected_edge = float(probability - 0.5)
                # ajuste por colas pesadas
                if abs(sig['skew']) > 2.0 or sig['kurtosis'] > 8.0:
                    expected_edge *= 0.5

                friction_cost = float((2.0 * self.s.taker_fee) + (self._last_market_quality.spread_bps / 10_000.0) + (self.s.slippage_bps / 10_000.0))
                decision, confidence, reason = self.decision_engine.decide(
                    probability=probability,
                    expected_edge=expected_edge,
                    friction_cost=friction_cost,
                    trading_enabled=not self.state.kill_switch,
                    market_operable=self._last_market_quality.operable,
                )
                self.state.last_block_reason = reason
                self.activity_text = reason

                cooldown = max(self.MIN_SECONDS_BETWEEN_TRADES - (now.timestamp() - self.last_trade_ts), 0.0) if self.last_trade_ts else 0.0
                if cooldown > 0 and decision in {'BUY', 'SELL'}:
                    decision = 'HOLD'
                    self.system_state = SystemState.COOLDOWN.value
                    self.state.last_block_reason = 'cooldown_active'
                    self.activity_text = f'COOLDOWN activo ({cooldown:.1f}s)'

                if decision in {'HOLD', 'DISABLE_TRADING'}:
                    if self.system_state not in {SystemState.COOLDOWN.value, 'KILL_SWITCH_ACTIVE'}:
                        self.system_state = SystemState.WAITING_EDGE.value

                if self.state.position_state == PositionState.FLAT and decision == 'SELL':
                    decision = 'HOLD'
                    self.system_state = SystemState.WAITING_EDGE.value
                    self.state.last_block_reason = 'no_position_to_sell'
                    self.activity_text = 'No hay posición para cerrar'

                if self.state.position_state == PositionState.LONG:
                    elapsed = (now - (self.state.position_opened_at or now)).total_seconds()
                    if self.state.tp_price > 0 and last_price >= self.state.tp_price:
                        decision = 'SELL'
                        self.activity_text = 'TP alcanzado'
                    elif self.state.sl_price > 0 and last_price <= self.state.sl_price:
                        decision = 'SELL'
                        self.activity_text = 'SL alcanzado'
                    elif elapsed >= max(float(self.s.target_scalp_seconds), 120.0):
                        decision = 'SELL'
                        self.activity_text = f'Cierre por tiempo razonable ({elapsed:.0f}s)'

                volatility = max(float(sig['volatility']), 1e-6)
                risk_fraction = self._risk_fraction(expected_edge=expected_edge, volatility=volatility)
                order_qty = 0.0
                if decision == 'BUY':
                    if risk_fraction <= 0.0:
                        decision = 'HOLD'
                        self.system_state = SystemState.WAITING_EDGE.value
                        self.state.last_block_reason = 'risk_fraction_zero'
                    else:
                        requested_notional = self.state.equity * risk_fraction
                        order_qty = requested_notional / max(last_price, 1e-9)
                elif decision == 'SELL':
                    order_qty = self.state.position_qty

                if decision in {'BUY', 'SELL'} and order_qty > 0.0:
                    self.system_state = SystemState.SENDING_ORDER.value
                    self.execution_engine.set_risk_context(
                        capital_total=self.state.equity,
                        risk_per_trade=risk_fraction if decision == 'BUY' else self.s.risk_per_trade,
                        signal_confidence=max(confidence, 0.1),
                    )
                    fill = await self._execute_order(decision, order_qty)
                    if fill and decision == 'BUY':
                        fill_price = float(fill['price'] or last_price)
                        self.state.position_state = PositionState.LONG
                        self.state.position_qty = float(fill['qty'])
                        self.state.entry_price = fill_price
                        self.state.position_opened_at = now
                        self.last_trade_ts = now.timestamp()
                        self.state.trades += 1
                        self.state.tp_price, self.state.sl_price = self.execution_engine.compute_dynamic_exit_levels(
                            entry_price=self.state.entry_price,
                            atr=float(sig['atr']),
                            side='BUY',
                        )
                        self.state.fees_paid += (self.state.position_qty * fill_price) * float(self.s.taker_fee)
                        self.system_state = SystemState.IN_POSITION.value
                        self.state.last_block_reason = 'none'
                    elif fill and decision == 'SELL':
                        fill_price = float(fill['price'] or last_price)
                        gross_pnl = (fill_price - self.state.entry_price) * self.state.position_qty
                        close_fee = (self.state.position_qty * fill_price) * float(self.s.taker_fee)
                        self.state.fees_paid += close_fee
                        net_pnl = gross_pnl - close_fee
                        self.state.realized_pnl += net_pnl
                        self.state.consecutive_losses = self.state.consecutive_losses + 1 if net_pnl <= 0 else 0
                        if net_pnl > 0:
                            self.state.winning_trades += 1
                        self.state.position_state = PositionState.FLAT
                        self.state.position_qty = 0.0
                        self.state.entry_price = 0.0
                        self.state.position_opened_at = None
                        self.state.tp_price = 0.0
                        self.state.sl_price = 0.0
                        self.state.unrealized_pnl = 0.0
                        self.state.capital_in_position = 0.0
                        self.last_trade_ts = now.timestamp()
                        self.system_state = SystemState.COOLDOWN.value
                        self.state.last_block_reason = 'none'
                    else:
                        self.system_state = SystemState.BLOCKED_BY_RISK.value

                self.state.consecutive_cycle_errors = 0
                self._publish_dashboard(
                    decision=decision,
                    confidence=confidence,
                    expected_edge=expected_edge,
                    mom=breakdown.momentum,
                    rev=breakdown.mean_reversion,
                    reg_prob=breakdown.regime,
                    regime=regime,
                    last_price=last_price,
                    binance_state='OK',
                )
                await asyncio.sleep(self.s.loop_interval_seconds)

            except Exception as exc:
                logger.error('kernel_error', error=str(exc))
                traceback.print_exc()
                should_stop = self._handle_cycle_exception(exc)
                if should_stop:
                    break
                await asyncio.sleep(1)

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with suppress(NotImplementedError):
                loop.add_signal_handler(sig, self.shutdown_event.set)
