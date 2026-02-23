from __future__ import annotations

import asyncio
import signal
import time
from collections import Counter, deque
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
from reco_trading.core.signal_fusion import SignalCombiner
from reco_trading.core.system_state import SystemState
from reco_trading.core.mean_reversion_model import MeanReversionModel
from reco_trading.core.momentum_model import MomentumModel
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
    pending_action: str | None = None


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
        self.last_confidence: float = 0.0
        self.last_scores: dict[str, float] = {}
        self.last_reason: str = 'booting'
        self._context: dict[str, Any] = {
            'momentum': 0.5,
            'reversion': 0.5,
            'global_probability': 0.5,
            'expected_edge': 0.0,
            'friction_cost': 0.0,
            'trading_enabled': True,
            'market_operable': True,
            'force_hold': False,
            'reason_prefix': '',
            'effective_min_edge': self.min_edge,
            'confidence_threshold': 0.0,
            'regime_uncertain': False,
        }

    def update_context(
        self,
        *,
        momentum: float,
        reversion: float,
        global_probability: float,
        expected_edge: float,
        friction_cost: float,
        trading_enabled: bool,
        market_operable: bool,
        force_hold: bool = False,
        reason_prefix: str = '',
        effective_min_edge: float | None = None,
        confidence_threshold: float = 0.0,
        regime_uncertain: bool = False,
    ) -> None:
        self._context = {
            'momentum': float(np.clip(momentum, 0.0, 1.0)),
            'reversion': float(np.clip(reversion, 0.0, 1.0)),
            'global_probability': float(np.clip(global_probability, 0.0, 1.0)),
            'expected_edge': float(expected_edge),
            'friction_cost': float(max(friction_cost, 0.0)),
            'trading_enabled': bool(trading_enabled),
            'market_operable': bool(market_operable),
            'force_hold': bool(force_hold),
            'reason_prefix': str(reason_prefix),
            'effective_min_edge': float(max(effective_min_edge if effective_min_edge is not None else self.min_edge, 0.0)),
            'confidence_threshold': float(np.clip(confidence_threshold, 0.0, 1.0)),
            'regime_uncertain': bool(regime_uncertain),
        }

    def decide(self) -> str:
        momentum = float(self._context['momentum'])
        reversion = float(self._context['reversion'])
        p = float(self._context['global_probability'])
        expected_edge = float(self._context['expected_edge'])
        friction_cost = float(self._context['friction_cost'])
        trading_enabled = bool(self._context['trading_enabled'])
        market_operable = bool(self._context['market_operable'])
        force_hold = bool(self._context['force_hold'])
        reason_prefix = str(self._context['reason_prefix'])
        effective_min_edge = float(self._context['effective_min_edge'])
        confidence_threshold = float(self._context['confidence_threshold'])
        regime_uncertain = bool(self._context['regime_uncertain'])

        self.last_scores = {
            'momentum': momentum,
            'reversion': reversion,
            'global': p,
            'effective_min_edge': effective_min_edge,
            'confidence_threshold': confidence_threshold,
        }
        self.last_confidence = p

        if not trading_enabled:
            self.last_reason = f'{reason_prefix}kill_switch_active'
            return 'DISABLE_TRADING'
        if force_hold:
            self.last_reason = f'{reason_prefix}learning_market'
            return 'HOLD'
        if not market_operable:
            self.last_reason = f'{reason_prefix}market_not_operable'
            return 'HOLD'
        if regime_uncertain:
            self.last_reason = f'{reason_prefix}regime_uncertain'
            return 'HOLD'
        if p < confidence_threshold:
            self.last_reason = f'{reason_prefix}confidence_below_threshold p={p:.4f} min={confidence_threshold:.4f}'
            return 'HOLD'
        if expected_edge <= friction_cost:
            self.last_reason = f'{reason_prefix}edge_below_friction edge={expected_edge:.6f} friction={friction_cost:.6f}'
            return 'HOLD'
        if abs(expected_edge) < effective_min_edge:
            self.last_reason = f'{reason_prefix}insufficient_edge edge={expected_edge:+.6f} min={effective_min_edge:.6f}'
            return 'HOLD'
        if expected_edge > 0:
            self.last_reason = f'{reason_prefix}positive_edge edge={expected_edge:+.6f}'
            return 'BUY'
        self.last_reason = f'{reason_prefix}negative_edge edge={expected_edge:+.6f}'
        return 'SELL'


class QuantKernel:
    MIN_SECONDS_BETWEEN_TRADES = 120
    MAX_CONSECUTIVE_CYCLE_ERRORS = 5
    MIN_WARMUP_SECONDS = 300
    MIN_WARMUP_BARS = 30

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
        self.operating_mode = self.s.operating_mode
        self.blocking_counters: Counter[str] = Counter()
        self.decision_audit_history: deque[dict[str, Any]] = deque(maxlen=self.s.decision_audit_history_size)
        self._daily_anchor_date = datetime.now(timezone.utc).date()
        self._daily_anchor_equity = 0.0

    @staticmethod
    def _timeframe_to_seconds(timeframe: str) -> int:
        normalized = str(timeframe).strip().lower()
        units = {'m': 60, 'h': 3600, 'd': 86_400}
        if len(normalized) < 2 or normalized[-1] not in units:
            return 60
        try:
            amount = int(normalized[:-1])
        except ValueError:
            return 60
        return max(amount * units[normalized[-1]], 60)

    async def initialize(self) -> None:
        timeframe_seconds = self._timeframe_to_seconds(self.s.timeframe)
        warmup_window_seconds = self.MIN_WARMUP_BARS * timeframe_seconds
        self.client = BinanceClient(
            self.s.binance_api_key.get_secret_value(),
            self.s.binance_api_secret.get_secret_value(),
            testnet=self.s.binance_testnet,
            confirm_mainnet=self.s.confirm_mainnet,
        )
        self.db = Database(self.s.postgres_dsn, self.s.postgres_admin_dsn)
        self.market_data = MarketDataService(self.client, self.s.symbol, self.s.timeframe)
        self.signal_engine = SignalEngine()
        self.data_buffer = DataBuffer(window_seconds=max(self.s.learning_phase_seconds, warmup_window_seconds))
        self.signal_combiner = SignalCombiner()
        self.decision_engine = DecisionEngine(min_edge=0.0040)
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
        self._daily_anchor_equity = free_usdt
        self._daily_anchor_date = datetime.now(timezone.utc).date()
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

    def _cooldown_seconds(self) -> float:
        timeframe_seconds = float(self._timeframe_to_seconds(self.s.timeframe))
        return max(float(self.MIN_SECONDS_BETWEEN_TRADES), timeframe_seconds)

    def _conservative_friction_cost(self, spread_bps: float, volatility: float) -> float:
        base_friction = (2.0 * self.s.taker_fee) + (spread_bps / 10_000.0) + (self.s.slippage_bps / 10_000.0)
        spread_penalty = max(spread_bps / 10_000.0, 0.0) * 0.5
        volatility_penalty = min(max(volatility / max(self.s.volatility_target, 1e-9), 0.0), 3.0) * 0.0006
        return float((base_friction * self.s.friction_safety_multiplier) + spread_penalty + volatility_penalty)

    def _minimum_operational_edge(self, *, effective_min_edge: float, friction_cost: float, regime: str) -> float:
        regime_buffer = 0.0015 if regime == 'HIGH_VOL' else (0.0008 if regime == 'RANGE' else 0.0004)
        return float(max(effective_min_edge, self.s.operational_edge_floor, (friction_cost * self.s.friction_safety_multiplier) + regime_buffer))

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
            today = datetime.now(timezone.utc).date()
            if today != self._daily_anchor_date:
                self._daily_anchor_date = today
                self._daily_anchor_equity = total_equity
            if self._daily_anchor_equity <= 0.0:
                self._daily_anchor_equity = total_equity
            daily_loss_ratio = (self._daily_anchor_equity - total_equity) / max(self._daily_anchor_equity, 1e-9)
            if daily_loss_ratio >= self.s.max_daily_loss:
                self.state.kill_switch = True
                return True, 'daily_loss_hard_stop'
        return False, 'none'

    def _is_warmup_complete(self, now_ts: float) -> tuple[bool, str]:
        if self.learning_started_at_ms is None:
            return False, 'warmup_missing_start'

        elapsed = max(now_ts - (float(self.learning_started_at_ms) / 1000.0), 0.0)
        bars = int(len(self.data_buffer.ohlcv))
        timeframe_seconds = self._timeframe_to_seconds(self.s.timeframe)
        max_bars_from_window = max(int(self.data_buffer.window_seconds / timeframe_seconds), 1)
        required_bars = max(20, min(self.MIN_WARMUP_BARS, max_bars_from_window))
        if elapsed < self.MIN_WARMUP_SECONDS or bars < required_bars:
            return False, f'warmup_active elapsed={elapsed:.1f}s bars={bars}/{required_bars}'
        return True, 'warmup_complete'

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
        logger.warning(
            "Execution blocked by firewall: reason=%s risk_snapshot=%s",
            reason,
            risk_snapshot
        )

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

    def _dynamic_thresholds(
        self,
        *,
        volatility: float,
        spread_bps: float,
        regime: str,
    ) -> tuple[float, float, bool]:
        timeframe_seconds = self._timeframe_to_seconds(self.s.timeframe)
        timeframe_factor = float(np.clip(timeframe_seconds / 300.0, 0.7, 1.4))
        vol_ratio = float(np.clip(volatility / max(self.s.volatility_target, 1e-6), 0.2, 3.0))
        spread_ratio = float(np.clip(spread_bps / max(self.s.market_max_spread_bps, 1e-6), 0.1, 2.0))

        edge_modifier = float(np.clip((1.1 - (0.25 * vol_ratio) - (0.10 * spread_ratio)) * timeframe_factor, 0.35, 1.4))
        confidence_modifier = float(np.clip(0.85 + (0.20 * vol_ratio) + (0.08 * spread_ratio), 0.75, 1.4))

        base_edge = self.decision_engine.min_edge
        confidence_base = self.s.confidence_hold_threshold
        effective_min_edge = float(max(base_edge * edge_modifier, self.s.minimal_mode_min_edge_floor))
        confidence_threshold = float(np.clip(confidence_base * confidence_modifier, 0.50, 0.95))

        regime_uncertain = regime == 'HIGH_VOL' and self.operating_mode != 'MINIMAL'
        if regime == 'HIGH_VOL' and self.operating_mode == 'MINIMAL':
            confidence_threshold = float(max(confidence_threshold, self.s.minimal_mode_regime_uncertain_floor))

        return effective_min_edge, confidence_threshold, regime_uncertain

    def _record_decision_audit(
        self,
        *,
        decision: str,
        expected_edge: float,
        effective_min_edge: float,
        confidence: float,
        confidence_threshold: float,
        friction_cost: float,
        regime_uncertain: bool,
        policy_block_reasons: list[str],
    ) -> None:
        blockers: list[str] = []
        if expected_edge <= friction_cost:
            blockers.append(f'EDGE_BELOW_FRICTION ({expected_edge:.6f} <= {friction_cost:.6f})')
        if abs(expected_edge) < effective_min_edge:
            blockers.append(f'EDGE_BELOW_THRESHOLD ({abs(expected_edge):.6f} < {effective_min_edge:.6f})')
        if confidence < confidence_threshold:
            blockers.append(f'CONFIDENCE_TOO_LOW ({confidence:.4f} < {confidence_threshold:.4f})')
        if regime_uncertain:
            blockers.append('REGIME_UNCERTAIN')
        blockers.extend(policy_block_reasons)

        final_decision = decision if decision in {'BUY', 'SELL'} else 'HOLD'
        payload = {
            'ts': datetime.now(timezone.utc).isoformat(),
            'decision': final_decision,
            'blocked_by': blockers,
        }
        self.decision_audit_history.append(payload)
        for blocker in blockers:
            key = blocker.split(' ', 1)[0]
            self.blocking_counters[key] += 1

        if blockers:
            logger.info('Decision: {} | Blocked by: {}', final_decision, blockers)
        else:
            logger.info('Decision: {} | Blocked by: none', final_decision)

    def _publish_dashboard(
        self,
        decision: str,
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
            cooldown = max(self._cooldown_seconds() - (now.timestamp() - self.last_trade_ts), 0.0)

        risk_state = 'BLOCKED' if self.should_block_trading() else 'OK'
        self.dashboard.update(
            VisualSnapshot(
                price=max(last_price, 0.0),
                equity=total_equity,
                pnl=self.state.realized_pnl + self.state.unrealized_pnl - self.state.fees_paid,
                decision=decision,
                confidence=self.decision_engine.last_confidence,
                scores=dict(self.decision_engine.last_scores),
                regime=regime,
                risk_state=risk_state,
                execution_state=self.execution_status,
                reason=self.decision_engine.last_reason,
            )
        )

    def _handle_cycle_exception(self, exc: Exception) -> bool:
        self.state.consecutive_cycle_errors += 1
        self.activity_text = f'Error de ciclo: {exc}'
        self.execution_status = 'ERROR'
        self.system_state = SystemState.ERROR.value
        self.decision_engine.update_context(
            momentum=0.5,
            reversion=0.5,
            global_probability=0.0,
            expected_edge=0.0,
            friction_cost=0.0,
            trading_enabled=False,
            market_operable=False,
            force_hold=True,
            reason_prefix='phase=error;',
        )
        self.decision_engine.decide()
        self._publish_dashboard('HOLD', 0.0, 0.5, 0.5, 0.5, 'ERROR', 0.0, 'ERROR')
        if self.state.consecutive_cycle_errors >= self.MAX_CONSECUTIVE_CYCLE_ERRORS:
            self._shutdown_reason = 'max_consecutive_cycle_errors'
            self.shutdown_event.set()
            return True
        return False

    async def run(self) -> None:
        try:
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
                                        self.decision_engine.update_context(
                                            momentum=0.5,
                                            reversion=0.5,
                                            global_probability=0.5,
                                            expected_edge=0.0,
                                            friction_cost=0.0,
                                            trading_enabled=not self.state.kill_switch,
                                            market_operable=False,
                                            force_hold=True,
                                            reason_prefix='phase=learning;',
                                        )
                                        decision = self.decision_engine.decide()
                                        self.state.last_block_reason = self.decision_engine.last_reason
                                        self.activity_text = f'APRENDIENDO MERCADO ({progress:.0%}) restante={remaining:.1f}s'
                                        self._publish_dashboard(decision, 0.0, 0.5, 0.5, 0.5, 'RANGE', float(ohlcv["close"].iloc[-1]), 'OK', remaining)
                                        await asyncio.sleep(self.s.loop_interval_seconds)
                                        continue

                                    if self.should_block_trading():
                                        self._publish_dashboard('DISABLE_TRADING', 0.0, 0.5, 0.5, 0.5, 'RANGE', float(ohlcv["close"].iloc[-1]), 'OK')
                                        await asyncio.sleep(self.s.loop_interval_seconds)
                                        continue

                                    self.system_state = SystemState.ANALYZING_MARKET.value
                                    if not self._last_market_quality.operable:
                                        self.system_state = SystemState.WAITING_EDGE.value
                                        self.state.last_block_reason = self._last_market_quality.reason
                                        self.activity_text = f'Mercado no operable: {self._last_market_quality.reason}'
                                        self._publish_dashboard('HOLD', 0.0, 0.5, 0.5, 0.5, 'RANGE', float(ohlcv["close"].iloc[-1]), 'OK')
                                        await asyncio.sleep(self.s.loop_interval_seconds)
                                        continue
                                    if self.learning_started_at_ms is None:
                                        self.learning_started_at_ms = int(ohlcv['timestamp'].iloc[0].timestamp() * 1000)

                                    spread_bps = await self.market_data.latest_spread_bps()
                                    self.data_buffer.record_spread(spread_bps)

                                    if self.data_buffer.in_learning_phase(self.learning_started_at_ms, now.timestamp()):
                                        self.system_state = SystemState.LEARNING_MARKET.value
                                        stats = self.data_buffer.learning_stats(ohlcv)
                                        self.activity_text = f'Aprendizaje 5m: vol={stats.rolling_volatility:.5f} atr={stats.atr:.2f} spread={stats.average_spread:.2f}bps'
                                        self.state.last_block_reason = 'learning_phase_active'
                                        self._publish_dashboard('HOLD', 0.0, 0.5, 0.5, 0.5, stats.dominant_regime, float(ohlcv['close'].iloc[-1]), 'OK')
                                        await asyncio.sleep(self.s.loop_interval_seconds)
                                        continue

                                    warmup_ready, warmup_reason = self._is_warmup_complete(now.timestamp())
                                    if not warmup_ready:
                                        self.system_state = SystemState.LEARNING_MARKET.value
                                        self.state.last_block_reason = warmup_reason
                                        self.activity_text = warmup_reason
                                        self._publish_dashboard('HOLD', 0.0, 0.5, 0.5, 0.5, 'RANGE', float(ohlcv['close'].iloc[-1]), 'OK')
                                        await asyncio.sleep(self.s.loop_interval_seconds)
                                        continue

                                    self.system_state = SystemState.ANALYZING_MARKET.value
                                    ok_balance, free_usdt, reason = await self._fetch_account_balance()
                                    if not ok_balance:
                                        self.system_state = SystemState.BLOCKED_BY_RISK.value
                                        self.state.last_block_reason = reason
                                        self.activity_text = f'Balance inválido: {reason}'
                                        self._publish_dashboard('DISABLE_TRADING', 0.0, 0.5, 0.5, 0.5, 'RANGE', float(ohlcv["close"].iloc[-1]), 'ERROR')
                                        await asyncio.sleep(self.s.loop_interval_seconds)
                                        continue

                                    self.state.equity = free_usdt
                                    sig = self.signal_engine.generate(ohlcv, spread_bps=spread_bps)
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

                                    friction_cost = self._conservative_friction_cost(self._last_market_quality.spread_bps, float(sig['volatility']))
                                    friction_cost = float((2.0 * self.s.taker_fee) + (self._last_market_quality.spread_bps / 10_000.0) + (self.s.slippage_bps / 10_000.0))
                                    effective_min_edge, confidence_threshold, regime_uncertain = self._dynamic_thresholds(
                                        volatility=float(sig['volatility']),
                                        spread_bps=self._last_market_quality.spread_bps,
                                        regime=regime,
                                    )
                                    self.decision_engine.update_context(
                                        momentum=float(breakdown.momentum),
                                        reversion=float(breakdown.mean_reversion),
                                        global_probability=probability,
                                        expected_edge=expected_edge,
                                        friction_cost=friction_cost,
                                        trading_enabled=not self.state.kill_switch,
                                        market_operable=self._last_market_quality.operable,
                                        effective_min_edge=effective_min_edge,
                                        confidence_threshold=confidence_threshold,
                                        regime_uncertain=regime_uncertain,
                                    )
                                    statistical_decision = self.decision_engine.decide()
                                    executable_decision = statistical_decision
                                    policy_block_reasons: list[str] = []
                                    self.state.pending_action = statistical_decision if statistical_decision in {'BUY', 'SELL'} else None
                                    self.state.last_block_reason = self.decision_engine.last_reason
                                    self.activity_text = self.decision_engine.last_reason

                                    cooldown = max(self._cooldown_seconds() - (now.timestamp() - self.last_trade_ts), 0.0) if self.last_trade_ts else 0.0
                                    cooldown = max(self.MIN_SECONDS_BETWEEN_TRADES - (now.timestamp() - self.last_trade_ts), 0.0) if self.last_trade_ts else 0.0
                                    if cooldown > 0 and executable_decision in {'BUY', 'SELL'}:
                                        executable_decision = 'HOLD'
                                        policy_block_reasons.append(f'COOLDOWN_ACTIVE ({cooldown:.1f}s)')
                                        self.system_state = SystemState.COOLDOWN.value
                                        self.state.last_block_reason = 'cooldown_active'
                                        self.activity_text = f'COOLDOWN activo ({cooldown:.1f}s)'

                                    if self.state.position_state == PositionState.FLAT and executable_decision == 'SELL':
                                        executable_decision = 'HOLD'
                                        policy_block_reasons.append('NO_POSITION_TO_SELL')
                                        self.system_state = SystemState.WAITING_EDGE.value
                                        self.state.last_block_reason = 'no_position_to_sell'
                                        self.activity_text = 'No hay posición para cerrar'

                                    if self.operating_mode == 'MINIMAL' and self.state.position_state == PositionState.LONG and executable_decision == 'BUY':
                                        executable_decision = 'HOLD'
                                        policy_block_reasons.append('MINIMAL_SINGLE_POSITION_ACTIVE')

                                    operational_edge_floor = self._minimum_operational_edge(
                                        effective_min_edge=effective_min_edge,
                                        friction_cost=friction_cost,
                                        regime=regime,
                                    )
                                    if executable_decision in {'BUY', 'SELL'} and abs(expected_edge) < operational_edge_floor:
                                        executable_decision = 'HOLD'
                                        policy_block_reasons.append(
                                            f'OPERATING_EDGE_TOO_LOW ({abs(expected_edge):.6f} < {operational_edge_floor:.6f})'
                                        )
                                    if executable_decision in {'BUY', 'SELL'} and self.decision_engine.last_confidence < max(confidence_threshold, 0.62):
                                        executable_decision = 'HOLD'
                                        policy_block_reasons.append(
                                            f'PAYOFF_CONFIDENCE_MISMATCH ({self.decision_engine.last_confidence:.4f} < {max(confidence_threshold, 0.62):.4f})'
                                        )

                                    if executable_decision in {'HOLD', 'DISABLE_TRADING'}:
                                        if self.system_state not in {SystemState.COOLDOWN.value, 'KILL_SWITCH_ACTIVE'}:
                                            self.system_state = SystemState.WAITING_EDGE.value

                                    decision = executable_decision

                                    if self.state.position_state == PositionState.LONG:
                                        elapsed = (now - (self.state.position_opened_at or now)).total_seconds()
                                        if self.state.tp_price > 0 and last_price >= self.state.tp_price:
                                            decision = 'SELL'
                                            self.activity_text = 'TP alcanzado'
                                        elif self.state.sl_price > 0 and last_price <= self.state.sl_price:
                                            decision = 'SELL'
                                            self.activity_text = 'SL alcanzado'
                                        elif elapsed >= max(float(self.s.target_scalp_seconds), self._cooldown_seconds()):
                                            open_pnl = (last_price - self.state.entry_price) * self.state.position_qty
                                            estimated_exit_fee = (self.state.position_qty * last_price) * float(self.s.taker_fee)
                                            if open_pnl > estimated_exit_fee:
                                                decision = 'SELL'
                                                self.activity_text = f'Cierre por tiempo con ganancia ({elapsed:.0f}s)'

                                    volatility = max(float(sig['volatility']), 1e-6)
                                    risk_fraction = self._risk_fraction(expected_edge=expected_edge, volatility=volatility)
                                    order_qty = 0.0
                                    if decision == 'BUY':
                                        if self.operating_mode == 'MINIMAL':
                                            minimal_notional = min(self.s.minimal_fixed_position_notional, max(self.state.equity * 0.25, 0.0))
                                            absolute_risk_cap = max(self.s.minimal_absolute_risk_usdt, 1.0)
                                            requested_notional = max(0.0, min(minimal_notional, absolute_risk_cap))
                                            if requested_notional < self.s.minimal_economic_notional:
                                                decision = 'HOLD'
                                                self.system_state = SystemState.WAITING_EDGE.value
                                                self.state.last_block_reason = 'minimal_economic_notional_not_met'
                                                self.activity_text = 'HOLD por notional económico insuficiente'
                                                policy_block_reasons.append(
                                                    f'MINIMAL_ECONOMIC_NOTIONAL ({requested_notional:.4f} < {self.s.minimal_economic_notional:.4f})'
                                                )
                                            elif requested_notional <= 0:
                                                decision = 'HOLD'
                                                self.system_state = SystemState.WAITING_EDGE.value
                                                self.state.last_block_reason = 'minimal_notional_zero'
                                                self.activity_text = 'HOLD por notional mínimo inválido'
                                                policy_block_reasons.append('MINIMAL_NOTIONAL_ZERO')
                                            else:
                                                order_qty = requested_notional / max(last_price, 1e-9)
                                        elif risk_fraction <= 0:
                                            decision = 'HOLD'
                                            self.system_state = SystemState.WAITING_EDGE.value
                                            self.state.last_block_reason = 'risk_fraction_zero'
                                            self.activity_text = 'HOLD por riesgo insuficiente para sizing'
                                            policy_block_reasons.append('RISK_FRACTION_ZERO')
                                        else:
                                            requested_notional = self.state.equity * risk_fraction
                                            order_qty = requested_notional / max(last_price, 1e-9)
                                    elif decision == 'SELL':
                                        order_qty = self.state.position_qty

                                    if decision in {'BUY', 'SELL'} and order_qty <= 0.0:
                                        policy_block_reasons.append('ORDER_QTY_ZERO_POLICY')
                                        decision = 'HOLD'

                                    self._record_decision_audit(
                                        decision=decision,
                                        expected_edge=expected_edge,
                                        effective_min_edge=effective_min_edge,
                                        confidence=self.decision_engine.last_confidence,
                                        confidence_threshold=confidence_threshold,
                                        friction_cost=friction_cost,
                                        regime_uncertain=regime_uncertain,
                                        policy_block_reasons=policy_block_reasons,
                                    )

                                    if decision in {'BUY', 'SELL'} and order_qty > 0.0:
                                        # --- MINIMAL MODE: enforce exchange minimums ---
                                        if self.s.operating_mode == 'MINIMAL':
                                            exchange_min_qty = await self.execution_engine._firewall._min_size(self.client, self.s.symbol)
                                            if exchange_min_qty is not None:
                                                if order_qty < exchange_min_qty:
                                                    logger.info(
                                                        "MINIMAL MODE: order_qty %.10f < exchange_min_qty %.10f, adjusting to minimum",
                                                        order_qty,
                                                        exchange_min_qty
                                                    )
                                                    order_qty = exchange_min_qty
                                        self.system_state = SystemState.SENDING_ORDER.value
                                        signal_confidence = max(self.decision_engine.last_confidence, 0.1)
                                        risk_context_per_trade = risk_fraction if decision == 'BUY' else self.s.risk_per_trade
                                        if decision == 'BUY' and self.s.operating_mode == 'MINIMAL':
                                            required_min_risk = (order_qty * last_price) / max(self.state.equity * signal_confidence, 1e-9)
                                            risk_context_per_trade = max(risk_context_per_trade, required_min_risk, self.s.risk_per_trade)
                                        self.execution_engine.set_risk_context(
                                            capital_total=self.state.equity,
                                            risk_per_trade=risk_context_per_trade,
                                            signal_confidence=signal_confidence,
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
                                            self.activity_text = f'BUY ejecutado qty={self.state.position_qty:.6f}'
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
                                        expected_edge=expected_edge,
                                        mom=breakdown.momentum,
                                        rev=breakdown.mean_reversion,
                                        reg_prob=breakdown.regime,
                                        regime=regime,
                                        last_price=last_price,
                                        binance_state='OK',
                                    )

                except Exception as exc:
                    logger.error(f'Kernel error: {exc}')
                    should_stop = self._handle_cycle_exception(exc)
                    if should_stop:
                        break
                    await asyncio.sleep(1)
        finally:
            await self._shutdown_resources()

    async def _shutdown_resources(self) -> None:
        with suppress(Exception):
            self.dashboard.stop()
        with suppress(Exception):
            self.metrics_exporter.stop()
        client = getattr(self, 'client', None)
        if client is not None:
            with suppress(Exception):
                await client.close()

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with suppress(NotImplementedError):
                loop.add_signal_handler(sig, self.shutdown_event.set)
