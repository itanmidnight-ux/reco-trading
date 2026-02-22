from __future__ import annotations

import asyncio
import signal
import traceback
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from reco_trading.config.settings import get_settings
from reco_trading.core.data_buffer import DataBuffer
from reco_trading.core.execution_engine import ExecutionEngine
from reco_trading.core.feature_engine import FeatureEngine
from reco_trading.core.market_data import MarketDataService
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
    daily_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    trades: int = 0
    winning_trades: int = 0
    rolling_returns: list[float] = field(default_factory=list)
    consecutive_cycle_errors: int = 0
    last_block_reason: str = 'booting'

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

    def generate(self, ohlcv: pd.DataFrame, spread_bps: float) -> dict[str, Any]:
        feats = self.feature_engine.build(ohlcv)
        if feats.empty:
            raise ValueError('Features vacías')

        snapshot = self.feature_engine.market_snapshot(ohlcv, spread=spread_bps)
        momentum = float(np.clip(self.momentum.predict_from_snapshot(snapshot), 0.0, 1.0))
        mean_rev = float(np.clip(self.reversion.predict_from_snapshot(snapshot), 0.0, 1.0))

        return {
            'snapshot': snapshot,
            'model_scores': {'momentum': momentum, 'mean_reversion': mean_rev},
            'atr': snapshot.atr,
            'returns': snapshot.returns,
            'prices': feats['close'].tail(300),
        }


class DecisionEngine:
    def __init__(self, min_edge: float = 0.08) -> None:
        self.min_edge = float(min_edge)

    def decide(self, probability: float, *, force_hold: bool = False, reason_prefix: str = '') -> tuple[str, float, str]:
        p = float(np.clip(probability, 0.0, 1.0))
        if force_hold:
            return 'HOLD', p, f'{reason_prefix}learning_market'

        edge = p - 0.5
        if abs(edge) < self.min_edge:
            return 'HOLD', p, f'{reason_prefix}insufficient_edge edge={edge:+.4f}'
        if edge > 0:
            return 'BUY', p, f'{reason_prefix}positive_edge edge={edge:+.4f}'
        return 'SELL', p, f'{reason_prefix}negative_edge edge={edge:+.4f}'


class QuantKernel:
    MIN_SECONDS_BETWEEN_TRADES = 10
    MAX_CONSECUTIVE_CYCLE_ERRORS = 5

    def __init__(self) -> None:
        self.s = get_settings()
        self.state = RuntimeState()
        self.last_trade_ts: float | None = None
        self.initial_equity = 0.0
        self._shutdown_reason = 'running'

        self.metrics = TradingMetrics()
        self.metrics_exporter = MetricsExporter(port=self.s.monitoring_metrics_port, addr=self.s.monitoring_metrics_host)
        self.dashboard = TerminalDashboard()
        self.shutdown_event = asyncio.Event()
        self.system_state = SystemState.WAITING_FOR_DATA.value
        self.activity_text = 'Inicializando kernel'
        self.learning_started_at_ms: int | None = None

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

    def should_block_trading(self) -> bool:
        if self.shutdown_event.is_set():
            return True
        if self.initial_equity <= 0:
            return False
        current_equity = self.state.equity + self.state.unrealized_pnl
        drawdown = 1.0 - (current_equity / max(self.initial_equity, 1e-9))
        if drawdown >= self.s.max_global_drawdown:
            self.state.last_block_reason = 'excessive_drawdown'
            self.system_state = SystemState.BLOCKED_BY_RISK.value
            return True
        return False

    def on_firewall_rejection(self, reason: str, risk_snapshot: dict[str, Any]) -> None:
        self.state.last_block_reason = f'firewall:{reason}'
        self.system_state = SystemState.BLOCKED_BY_RISK.value
        logger.warning('Execution blocked by firewall', reason=reason, risk_snapshot=risk_snapshot)

    async def _execute_order(self, side: str, qty: float) -> dict[str, Any] | None:
        if qty <= 0.0:
            self.state.last_block_reason = 'invalid_qty'
            return None
        fill = await self.execution_engine.execute(side, qty)
        if not fill:
            self.state.last_block_reason = self.execution_engine.last_rejection_reason or 'execution_returned_none'
            return None
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

    def _confidence_to_risk_fraction(self, confidence: float) -> float:
        c = float(confidence)
        s = self.s
        if c < s.confidence_hold_threshold:
            return 0.0
        if c >= s.confidence_tier_4:
            return min(s.confidence_alloc_tier_4, s.max_confidence_allocation)
        if c >= s.confidence_tier_3:
            return min(s.confidence_alloc_tier_3, s.max_confidence_allocation)
        if c >= s.confidence_tier_2:
            return min(s.confidence_alloc_tier_2, s.max_confidence_allocation)
        if c >= s.confidence_tier_1:
            return min(s.confidence_alloc_tier_1, s.max_confidence_allocation)
        return 0.0

    def _publish_dashboard(
        self,
        decision: str,
        confidence: float,
        mom: float,
        rev: float,
        reg_prob: float,
        regime: str,
        last_price: float,
        binance_state: str,
        learning_remaining_seconds: float = 0.0,
    ) -> None:
        total_equity = self.state.equity + self.state.unrealized_pnl
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
                pnl_total=self.state.daily_pnl + self.state.unrealized_pnl,
                pnl_diario=self.state.daily_pnl,
                drawdown=drawdown,
                riesgo_activo=float(np.clip(self.s.risk_per_trade, 0.0, 1.0)),
                exposicion=max(self.state.position_qty * max(last_price, 0.0), 0.0),
                trades=self.state.trades,
                win_rate=(self.state.winning_trades / self.state.trades if self.state.trades else 0.0),
                expectancy=confidence - 0.5,
                sharpe_rolling=0.0,
                regimen=regime,
                senal=decision,
                latencia_ms=0.0,
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
            )
        )

    def _handle_cycle_exception(self, exc: Exception) -> bool:
        self.state.consecutive_cycle_errors += 1
        self.activity_text = f'Error de ciclo: {exc}'
        self.system_state = SystemState.ERROR.value
        self._publish_dashboard('HOLD', 0.0, 0.5, 0.5, 0.5, 'ERROR', 0.0, 'ERROR')
        if self.state.consecutive_cycle_errors >= self.MAX_CONSECUTIVE_CYCLE_ERRORS:
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
                learning_progress, learning_remaining = self.data_buffer.learning_progress(self.learning_started_at_ms, now.timestamp())

                if learning_remaining > 0:
                    self.system_state = SystemState.LEARNING_MARKET.value
                    stats = self.data_buffer.learning_stats()
                    decision, confidence, reason = self.decision_engine.decide(0.5, force_hold=True, reason_prefix='phase=learning;')
                    self.state.last_block_reason = reason
                    self.activity_text = (
                        f'APRENDIENDO MERCADO ({learning_progress:.0%}) '\
                        f'restante={learning_remaining:.1f}s vol={stats.rolling_volatility:.5f} atr={stats.atr:.2f} spread={stats.average_spread:.2f}bps'
                    )
                    self._publish_dashboard(
                        decision,
                        confidence,
                        0.5,
                        0.5,
                        0.5,
                        stats.dominant_regime,
                        float(ohlcv['close'].iloc[-1]),
                        'OK',
                        learning_remaining_seconds=learning_remaining,
                    )
                    await asyncio.sleep(self.s.loop_interval_seconds)
                    continue

                self.system_state = SystemState.ANALYZING_MARKET.value
                ok_balance, free_usdt, reason = await self._fetch_account_balance()
                if not ok_balance:
                    self.system_state = SystemState.BLOCKED_BY_RISK.value
                    self.activity_text = f'Sin operación: balance inválido ({reason})'
                    self.state.last_block_reason = reason
                    self._publish_dashboard('HOLD', 0.0, 0.5, 0.5, 0.5, 'UNKNOWN', float(ohlcv['close'].iloc[-1]), 'ERROR')
                    await asyncio.sleep(self.s.loop_interval_seconds)
                    continue

                self.state.equity = free_usdt + max(self.state.daily_pnl, 0.0)
                sig = self.signal_engine.generate(self.data_buffer.ohlcv, spread_bps=spread_bps)
                last_price = float(sig['prices'].iloc[-1])

                if self.state.position_state == PositionState.LONG:
                    self.state.unrealized_pnl = (last_price - self.state.entry_price) * self.state.position_qty
                else:
                    self.state.unrealized_pnl = 0.0

                if self.should_block_trading():
                    self.activity_text = f'Trading bloqueado por riesgo: {self.state.last_block_reason}'
                    self._publish_dashboard('HOLD', 0.0, 0.5, 0.5, 0.5, 'RISK', last_price, 'OK')
                    await asyncio.sleep(self.s.loop_interval_seconds)
                    continue

                regime_raw = self.regime_detector.predict(sig['returns'], sig['prices']).get('regime', 'range')
                regime = self._map_regime(regime_raw)
                reg_prob = 0.80 if regime == 'TREND' else (0.70 if regime == 'RANGE' else 0.60)
                breakdown = self.signal_combiner.combine(
                    sig['model_scores']['momentum'],
                    sig['model_scores']['mean_reversion'],
                    reg_prob,
                    regime,
                )

                decision, confidence, reason = self.decision_engine.decide(breakdown.combined)
                self.state.last_block_reason = reason
                self.activity_text = reason

                cooldown_remaining = 0.0
                if self.last_trade_ts:
                    cooldown_remaining = max(self.MIN_SECONDS_BETWEEN_TRADES - (now.timestamp() - self.last_trade_ts), 0.0)
                if cooldown_remaining > 0:
                    self.system_state = SystemState.COOLDOWN.value
                    decision = 'HOLD'
                    self.state.last_block_reason = 'cooldown_active'
                    self.activity_text = f'COOLDOWN activo, restante={cooldown_remaining:.1f}s'

                if decision == 'HOLD' and self.system_state != SystemState.COOLDOWN.value:
                    self.system_state = SystemState.WAITING_EDGE.value

                if self.state.position_state == PositionState.FLAT and decision == 'SELL':
                    decision = 'HOLD'
                    self.state.last_block_reason = 'no_position_to_sell'
                    self.activity_text = 'Sin posición abierta para cerrar'
                    self.system_state = SystemState.WAITING_EDGE.value

                if self.state.position_state == PositionState.LONG:
                    elapsed = (now - (self.state.position_opened_at or now)).total_seconds()
                    if self.state.tp_price > 0 and last_price >= self.state.tp_price:
                        decision = 'SELL'
                        confidence = 1.0
                        self.activity_text = f'Cierre por TP dinámico ({last_price:.2f} >= {self.state.tp_price:.2f})'
                    elif self.state.sl_price > 0 and last_price <= self.state.sl_price:
                        decision = 'SELL'
                        confidence = 1.0
                        self.activity_text = f'Cierre por SL dinámico ({last_price:.2f} <= {self.state.sl_price:.2f})'
                    elif elapsed >= self.s.target_scalp_seconds:
                        decision = 'SELL'
                        confidence = 1.0
                        self.activity_text = f'Cierre por scalping ({elapsed:.1f}s)'
                    elif elapsed >= self.s.max_position_seconds:
                        decision = 'SELL'
                        confidence = 1.0
                        self.activity_text = f'Cierre forzado por tiempo ({elapsed:.1f}s)'

                risk_fraction = self._confidence_to_risk_fraction(confidence)
                order_qty = 0.0
                if decision == 'BUY':
                    if risk_fraction <= 0.0:
                        decision = 'HOLD'
                        self.system_state = SystemState.WAITING_EDGE.value
                        self.state.last_block_reason = 'confidence_below_threshold'
                        self.activity_text = 'HOLD por confianza insuficiente para position sizing'
                    else:
                        requested_notional = self.state.equity * risk_fraction
                        order_qty = requested_notional / max(last_price, 1e-9)
                elif decision == 'SELL':
                    order_qty = self.state.position_qty

                if decision in {'BUY', 'SELL'} and order_qty > 0:
                    self.execution_engine.set_risk_context(
                        capital_total=self.state.equity,
                        risk_per_trade=risk_fraction if decision == 'BUY' else self.s.risk_per_trade,
                        signal_confidence=confidence,
                    )
                    self.system_state = SystemState.SENDING_ORDER.value
                    fill = await self._execute_order(decision, order_qty)

                    if fill and decision == 'BUY':
                        self.state.position_state = PositionState.LONG
                        self.state.position_qty = float(fill['qty'])
                        self.state.entry_price = float(fill['price'] or last_price)
                        self.state.position_opened_at = now
                        self.last_trade_ts = now.timestamp()
                        self.state.trades += 1
                        self.state.tp_price, self.state.sl_price = self.execution_engine.compute_dynamic_exit_levels(
                            entry_price=self.state.entry_price,
                            atr=float(sig['atr']),
                            side='BUY',
                        )
                        self.system_state = SystemState.IN_POSITION.value
                        self.activity_text = (
                            f'BUY ejecutado qty={self.state.position_qty:.6f} '
                            f'TP={self.state.tp_price:.2f} SL={self.state.sl_price:.2f}'
                        )
                        self.state.last_block_reason = 'none'
                    elif fill and decision == 'SELL':
                        pnl = self.state.unrealized_pnl
                        self.state.daily_pnl += pnl
                        self.state.equity += pnl
                        if pnl > 0:
                            self.state.winning_trades += 1
                        self.state.position_state = PositionState.FLAT
                        self.state.position_qty = 0.0
                        self.state.entry_price = 0.0
                        self.state.position_opened_at = None
                        self.state.tp_price = 0.0
                        self.state.sl_price = 0.0
                        self.state.unrealized_pnl = 0.0
                        self.last_trade_ts = now.timestamp()
                        self.system_state = SystemState.COOLDOWN.value
                        self.activity_text = f'SELL ejecutado, PnL={pnl:+.2f} USDT'
                        self.state.last_block_reason = 'none'
                    else:
                        self.system_state = SystemState.BLOCKED_BY_RISK.value
                        self.activity_text = f'Orden rechazada: {self.state.last_block_reason}'

                self.state.consecutive_cycle_errors = 0
                self._publish_dashboard(
                    decision,
                    confidence,
                    breakdown.momentum,
                    breakdown.mean_reversion,
                    breakdown.regime,
                    regime,
                    last_price,
                    'OK',
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
