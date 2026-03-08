from __future__ import annotations

import asyncio
import signal
import time
import uuid
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
from reco_trading.core.market_data import MarketDataService, MarketQuality, MarketQualityContract
from reco_trading.core.institutional_risk_manager import InstitutionalRiskManager, RiskLimits
from reco_trading.core.market_regime import MarketRegimeDetector
from reco_trading.core.signal_fusion import SignalCombiner
from reco_trading.core.system_state import SystemState
from reco_trading.core.mean_reversion_model import MeanReversionModel
from reco_trading.core.momentum_model import MomentumModel
from reco_trading.execution.execution_firewall import ExecutionFirewall
from reco_trading.execution.capital_protection_controller import CapitalProtectionController
from reco_trading.adaptive.frequency_controller import FrequencyController
from reco_trading.kernel.conditional_performance import ConditionalPerformanceTracker
from reco_trading.kernel.edge_monitor import EdgeMonitor, EdgeSnapshot
from reco_trading.kernel.regime_controller import RegimeController
from reco_trading.kernel.risk_of_ruin import RiskOfRuinEstimator, RiskOfRuinSnapshot
from reco_trading.kernel.capital_governor import CapitalGovernor
from reco_trading.portfolio.allocator import PortfolioAllocator
from reco_trading.portfolio.exposure_model import ExposureModel
from reco_trading.statistics.drift_detector import CUSUMDrift, DriftSignal, KLDivergenceDrift
from reco_trading.infra.binance_client import BinanceClient
from reco_trading.infra.database import Database
from reco_trading.hft.adaptive_market_maker import AdaptiveMarketMaker, MarketMakingState
from reco_trading.monitoring.metrics import MetricsExporter, TradingMetrics
from reco_trading.ui.terminal_dashboard import TerminalDashboard
from reco_trading.ui.visual_snapshot import VisualSnapshot


class PositionState(Enum):
    FLAT = 'flat'
    LONG = 'long'


@dataclass(slots=True)
class RuntimeState:
    equity: float = 0.0
    exchange_equity: float = 0.0
    realized_pnl: float = 0.0
    lifetime_realized_pnl: float = 0.0
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
    last_kill_switch_trigger: str | None = None
    avg_latency_ms: float = 0.0

    position_state: PositionState = PositionState.FLAT
    position_qty: float = 0.0
    entry_price: float = 0.0
    position_opened_at: datetime | None = None
    tp_price: float = 0.0
    sl_price: float = 0.0
    pending_action: str | None = None
    partial_exits_done: set[int] = field(default_factory=set)
    negative_edge_streak: int = 0
    binance_min_notional: float = 0.0
    final_order_notional: float = 0.0


@dataclass(slots=True)
class StrategyVote:
    strategy: str
    decision: str
    confidence: float
    reason: str



class SignalEngine:
    def __init__(self) -> None:
        self.feature_engine = FeatureEngine()
        self.momentum = MomentumModel()
        self.reversion = MeanReversionModel()
        self._normal = NormalDist(mu=0.0, sigma=1.0)
        self._zscore_cache: dict[str, float] = {}
        self._confidence_history: deque[float] = deque(maxlen=240)
        self._edge_history: deque[float] = deque(maxlen=240)

    def _dynamic_zscore(self, value: float, series: pd.Series, key: str, clip: float = 6.0) -> float:
        arr = pd.Series(series).dropna().astype(float)
        if arr.empty:
            return float(np.clip(value, -clip, clip))
        mu = float(arr.ewm(span=min(max(len(arr), 6), 80), adjust=False).mean().iloc[-1])
        sigma = float(arr.ewm(span=min(max(len(arr), 6), 80), adjust=False).std().iloc[-1] or 1e-9)
        z_value = float(np.clip((value - mu) / max(sigma, 1e-9), -clip, clip))
        self._zscore_cache[key] = z_value
        return z_value

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

        # normalización dinámica por z-score con reescalado por volatilidad
        edge_z = self._dynamic_zscore(mu, returns, key='drift', clip=8.0)
        edge_z = float(np.clip(edge_z / max(rolling_vol / max(sigma, 1e-9), 0.25), -8.0, 8.0))
        p_drift_up = float(self._normal.cdf(edge_z))

        # mean reversion estadístico: distancia a VWAP normalizada por volatilidad
        reversion_z = self._dynamic_zscore(-snapshot.vwap_distance, returns.tail(120), key='reversion', clip=8.0)
        reversion_z = float(np.clip(reversion_z / max(rolling_vol / max(sigma, 1e-9), 0.25), -8.0, 8.0))
        p_reversion_up = float(self._normal.cdf(reversion_z))

        # combinación entre señal estadística y modelo entrenado
        momentum_prob = float(np.clip(0.65 * p_drift_up + 0.35 * model_momentum, 0.0, 1.0))
        reversion_prob = float(np.clip(0.65 * p_reversion_up + 0.35 * model_reversion, 0.0, 1.0))

        correlation_penalty = float(np.clip(1.0 - abs(momentum_prob - reversion_prob), 0.0, 1.0))
        noise_penalty = float(np.clip((abs(skew) / 4.0) + max(kurtosis - 3.0, 0.0) / 12.0, 0.0, 0.55))
        stability_weight = float(np.clip(1.0 - np.std(returns.tail(20)) / max(np.std(returns.tail(120)), 1e-9), 0.25, 1.0))
        confidence_score = float(np.clip(((abs(momentum_prob - 0.5) + abs(reversion_prob - 0.5)) * stability_weight) - (0.20 * noise_penalty), 0.0, 1.0))
        self._confidence_history.append(confidence_score)

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
            'signal_vector': {
                'confidence_score': confidence_score,
                'stability_weight': stability_weight,
                'regime_alignment': float(np.clip(1.0 - correlation_penalty, 0.0, 1.0)),
                'volatility_adjusted_edge': float(np.clip((momentum_prob - reversion_prob) / max(rolling_vol, 1e-6), -1.0, 1.0)),
                'expected_value_net_costs': 0.0,
                'noise_penalty': noise_penalty,
                'correlation_penalty': correlation_penalty,
            },
        }


class DecisionEngine:
    def __init__(self, min_edge: float = 0.08, buy_threshold: float = 0.56, sell_threshold: float = 0.44) -> None:
        self.min_edge = float(min_edge)
        self.buy_threshold = float(np.clip(buy_threshold, 0.0, 1.0))
        self.sell_threshold = float(np.clip(sell_threshold, 0.0, 1.0))
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

    def decide(self, scores: dict[str, float] | None = None, regime: str | None = None) -> str | tuple[str, float]:
        if scores is not None:
            momentum = float(np.clip(scores.get('momentum', 0.5), 0.0, 1.0))
            reversion = float(np.clip(scores.get('mean_reversion', 0.5), 0.0, 1.0))
            p = float(np.clip((momentum + reversion) / 2.0, 0.0, 1.0))
            expected_edge = float(p - 0.5)
            regime_name = str((regime or 'range')).lower()
            tradable = regime_name != 'blocked'
            if regime_name == 'range':
                expected_edge *= 0.5
            if p >= self.buy_threshold and expected_edge >= self.min_edge and tradable:
                decision = 'BUY'
                reason = 'legacy_buy_threshold'
            elif p <= self.sell_threshold and abs(expected_edge) >= self.min_edge and tradable:
                decision = 'SELL'
                reason = 'legacy_sell_threshold'
            else:
                decision = 'HOLD'
                reason = 'legacy_threshold_guard'
            self.last_confidence = p
            self.last_scores = {'momentum': momentum, 'reversion': reversion, 'global': p}
            self.last_reason = reason
            return decision, p
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
        self.peak_equity = 0.0
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
        self._trade_returns: deque[float] = deque(maxlen=300)
        self._trade_pnls: deque[float] = deque(maxlen=300)
        self._rolling_stats: dict[str, float] = {
            'expectancy': 0.0,
            'winrate': 0.0,
            'profit_factor': 0.0,
            'sharpe': 0.0,
            'drawdown_pressure': 0.0,
        }
        self._startup_safety_until_monotonic: float = 0.0
        self._signal_quality: dict[str, float] = {
            'confidence_score': 0.0,
            'stability_weight': 0.0,
            'regime_alignment': 0.0,
            'volatility_adjusted_edge': 0.0,
            'expected_value_net_costs': 0.0,
        }
        self._latest_edge_snapshot = EdgeSnapshot(0.5, 0.0, 1.0, 0.5, 'INSUFFICIENT_DATA', 0.0, 0.0)
        self._latest_ruin_snapshot = RiskOfRuinSnapshot(1.0, 0.15)
        self._latest_regime_snapshot = {'current_regime': 'LOW_VOL_REGIME', 'regime_stability_score': 0.0}
        self._current_extended_regime = 'LOW_VOL_REGIME'
        self.portfolio_allocator = PortfolioAllocator()
        self.exposure_model = ExposureModel(lookback_bars=100)
        self.cusum_drift = CUSUMDrift()
        self.kl_drift = KLDivergenceDrift()
        self.frequency_controller = FrequencyController()
        self._asset_price_history: dict[str, pd.Series] = {}
        self._asset_edges: dict[str, float] = {}
        self._asset_volatility: dict[str, float] = {}
        self._portfolio_weights: dict[str, float] = {self.s.symbol: 1.0}
        self._latest_drift_signal = DriftSignal(drift_score=0.0, regime_change_probability=0.0)
        self._edge_scale_samples: deque[float] = deque(maxlen=200)
        self._context: dict[str, Any] = {
            'expected_edge': 0.0,
            'friction_cost': 0.0,
            'risk_fraction': float(self.s.risk_per_trade),
            'portfolio_weight': 1.0,
            'drift_score': 0.0,
        }
        self.market_making_engine = AdaptiveMarketMaker(MarketMakingState())
        self.arbitrage_engine_enabled = bool(self.s.enable_multi_exchange_arbitrage)

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
        timeframe = getattr(getattr(self, 's', None), 'timeframe', '5m')
        timeframe_seconds = self._timeframe_to_seconds(timeframe)
        warmup_window_seconds = self.MIN_WARMUP_BARS * timeframe_seconds
        self.client = BinanceClient(
            self.s.binance_api_key.get_secret_value(),
            self.s.binance_api_secret.get_secret_value(),
            testnet=self.s.binance_testnet,
            confirm_mainnet=self.s.confirm_mainnet,
        )
        logger.info('Startup trading mode: {}', 'TESTNET' if self.s.binance_testnet else 'PRODUCTION')
        self.db = Database(self.s.postgres_dsn, self.s.postgres_admin_dsn)
        await self.db.init()
        self.market_data = MarketDataService(self.client, self.s.symbol, self.s.timeframe)
        self.signal_engine = SignalEngine()
        self.data_buffer = DataBuffer(window_seconds=max(self.s.learning_phase_seconds, warmup_window_seconds))
        self.signal_combiner = SignalCombiner()
        self.decision_engine = DecisionEngine(min_edge=0.0040)
        self.regime_detector = MarketRegimeDetector(n_states=3)
        self.edge_monitor = EdgeMonitor(window=self.s.edge_monitor_window)
        self.regime_controller = RegimeController(window=self.s.edge_monitor_window)
        self.risk_of_ruin = RiskOfRuinEstimator()
        self.conditional_performance = ConditionalPerformanceTracker(window=self.s.conditional_performance_window)

        self.capital_governor = CapitalGovernor(hard_cap_global=10_000_000.0)
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
            capital_governor=self.capital_governor,
        )
        self.institutional_risk_manager = InstitutionalRiskManager(
            limits=RiskLimits(
                risk_per_trade=float(self.s.risk_per_trade),
                max_daily_loss=float(self.s.max_daily_loss),
                max_global_drawdown=float(self.s.max_global_drawdown),
                max_total_exposure=float(self.s.max_total_exposure),
                max_asset_exposure=float(self.s.max_asset_exposure),
                correlation_threshold=float(self.s.correlation_threshold),
            ),
            atr_multiplier=float(self.s.atr_stop_multiplier),
            capital_governor=self.capital_governor,
        )
        self.capital_protection_controller = CapitalProtectionController(
            client=self.client,
            order_service=self.execution_engine.order_service,
        )

        await self.client.ping()
        ok, free_usdt, exchange_equity, _, reason = await self._fetch_account_balance()
        if not ok:
            raise RuntimeError(f'Balance inicial inválido: {reason}')
        self.state.equity = free_usdt
        self.state.exchange_equity = exchange_equity
        self.initial_equity = exchange_equity
        self.peak_equity = exchange_equity
        self._daily_anchor_equity = exchange_equity
        self._daily_anchor_date = datetime.now(timezone.utc).date()
        self.state.last_block_reason = 'none'
        logger.info(f'Balance REAL sincronizado: free={free_usdt:.2f} total={exchange_equity:.2f} USDT')

    async def _fetch_account_balance(self) -> tuple[bool, float, float, float, str]:
        try:
            balance = await self.client.fetch_balance()
        except Exception as exc:
            return False, 0.0, 0.0, 0.0, f'binance_balance_error:{exc}'
        usdt_bucket = balance.get('USDT') if isinstance(balance, dict) else {}
        free_usdt = float((usdt_bucket or {}).get('free') or 0.0)
        used_usdt = float((usdt_bucket or {}).get('used') or 0.0)
        total_usdt = float((usdt_bucket or {}).get('total') or (free_usdt + used_usdt))
        base_asset = self.s.symbol.split('/', 1)[0]
        base_bucket = balance.get(base_asset) if isinstance(balance, dict) else {}
        base_qty = float((base_bucket or {}).get('free') or 0.0) + float((base_bucket or {}).get('used') or 0.0)
        try:
            ticker = await self.client.fetch_ticker(self.s.symbol)
            last_price = float(ticker.get('last') or ticker.get('close') or 0.0)
        except Exception:
            last_price = 0.0
        base_nav = max(base_qty, 0.0) * max(last_price, 0.0)
        total_nav = total_usdt + base_nav
        if not np.isfinite(total_nav) or total_nav <= 0.0:
            return False, 0.0, 0.0, 0.0, 'invalid_usdt_total_balance'
        if not np.isfinite(free_usdt) or free_usdt < 0.0:
            return False, 0.0, 0.0, 0.0, 'invalid_usdt_free_balance'
        if not np.isfinite(used_usdt) or used_usdt < 0.0:
            return False, 0.0, 0.0, 0.0, 'invalid_usdt_used_balance'
        return True, free_usdt, total_nav, used_usdt, 'ok'

    async def _restore_financial_state_from_db(self) -> None:
        pnl_snapshot = await self.db.get_realized_pnl_from_fills(symbol=self.s.symbol)
        realized = float(pnl_snapshot.get('realized_pnl') or 0.0)
        fees_paid = float(pnl_snapshot.get('fees_paid') or 0.0)
        # Historical fills are tracked for audit purposes only.
        # Session PnL must always start at 0 so dashboard PnL reflects
        # changes observed after startup reconciliation against exchange equity.
        self.state.lifetime_realized_pnl = realized if np.isfinite(realized) else 0.0
        self.state.realized_pnl = 0.0
        self.state.fees_paid = fees_paid if np.isfinite(fees_paid) and fees_paid >= 0.0 else 0.0

    async def _startup_reconciliation(self) -> None:
        ok, free_usdt, exchange_equity, _, reason = await self._fetch_account_balance()
        if not ok:
            raise RuntimeError(f'Balance de reconciliación inválido: {reason}')
        self.state.equity = free_usdt
        self.state.exchange_equity = exchange_equity

        service = self.execution_engine.order_service
        await service.load_from_ledger()
        await service.start()
        try:
            active = await self.db.load_active_idempotency_entries()
        except Exception:
            active = []
        open_orders = await self.client.fetch_open_orders(self.s.symbol)
        active_client_order_ids = {
            str(row.get('client_order_id') or '')
            for row in active
            if str(row.get('client_order_id') or '')
        }
        await self.db.cleanup_stale_reservations(
            open_orders=open_orders,
            active_client_order_ids=active_client_order_ids,
        )
        open_ids = {
            str(item.get('clientOrderId') or item.get('client_order_id') or '')
            for item in (open_orders or [])
        }
        for row in active:
            cid = str(row.get('client_order_id') or '')
            if not cid:
                continue
            if cid in open_ids:
                await self.db.update_idempotency_status(cid, 'SUBMITTED', str(row.get('exchange_order_id') or ''))
                continue
            found = await self.client.fetch_order_by_client_order_id(self.s.symbol, cid)
            if not found:
                continue
            status = str(found.get('status') or '').lower()
            exchange_order_id = str(found.get('id') or row.get('exchange_order_id') or '')
            if status in {'closed', 'filled'}:
                await self.db.update_idempotency_status(cid, 'FILLED', exchange_order_id)
            elif status in {'partially_filled'}:
                await self.db.update_idempotency_status(cid, 'PARTIALLY_FILLED', exchange_order_id)
            elif status in {'cancelled', 'canceled'}:
                await self.db.update_idempotency_status(cid, 'CANCELLED', exchange_order_id)
            else:
                await self.db.update_idempotency_status(cid, 'SUBMITTED', exchange_order_id)

        try:
            exchange_trades = await self.client.fetch_my_trades(self.s.symbol, limit=1000)
        except Exception as exc:
            logger.warning('startup_fetch_my_trades_failed: %s', exc)
            exchange_trades = []
        if exchange_trades:
            await self.db.reconcile_exchange_fills(self.s.symbol, exchange_trades)

        await self._restore_financial_state_from_db()
        snapshot = await self.db.restore_position_snapshot(self.s.symbol)
        net_qty = float(snapshot.get('net_qty') or 0.0)
        avg_entry = float(snapshot.get('avg_entry') or 0.0)

        base_asset = self.s.symbol.split('/', 1)[0]
        exchange_qty = 0.0
        try:
            position = await self.client.fetch_positions(self.s.symbol)
            exchange_qty = float(position.get('qty') or 0.0)
        except Exception as exc:
            logger.warning('startup_exchange_balance_reconciliation_failed: %s', exc)

        discrepancy_detected = abs(exchange_qty - net_qty) > 1e-8
        if discrepancy_detected:
            logger.warning(
                'startup_position_discrepancy_detected symbol=%s db_qty=%.10f exchange_qty=%.10f; applying exchange truth',
                self.s.symbol,
                net_qty,
                exchange_qty,
            )
            net_qty = max(exchange_qty, 0.0)

        # If exchange is the source of truth for qty, stale DB entry can produce
        # phantom unrealized PnL. Re-anchor entry to current market price.
        if net_qty > 0.0 and (avg_entry <= 0.0 or discrepancy_detected):
            try:
                ticker = await self.client.fetch_ticker(self.s.symbol)
                avg_entry = float(ticker.get('last') or ticker.get('close') or 0.0)
            except Exception:
                avg_entry = 0.0

        if net_qty > 0.0:
            self.state.position_state = PositionState.LONG
            self.state.position_qty = net_qty
            self.state.entry_price = avg_entry
            self.state.capital_in_position = net_qty * max(avg_entry, 0.0)
        else:
            self.state.position_state = PositionState.FLAT
            self.state.position_qty = 0.0
            self.state.entry_price = 0.0
            self.state.capital_in_position = 0.0

        self._repair_state_consistency(reference_price=float(avg_entry if avg_entry > 0.0 else 0.0))
        valid_financials, invariant_reason = self._validate_financial_invariants()
        if not valid_financials:
            raise RuntimeError(f'Financial invariant violation at startup: {invariant_reason}')

        total_equity = self._compute_total_equity()
        self._daily_anchor_equity = total_equity
        self._daily_anchor_date = datetime.now(timezone.utc).date()
        self.initial_equity = total_equity
        self.peak_equity = total_equity

    def _map_regime(self, raw_regime: str) -> str:
        r = str(raw_regime).lower()
        if r == 'trend':
            return 'TREND'
        if r in {'range', 'low_volatility'}:
            return 'RANGE'
        if r in {'high_volatility', 'volatile'}:
            return 'HIGH_VOL'
        return 'RANGE'

    def _regime_probability(self, *, mapped_regime: str, detector_confidence: float, stability_score: float = 0.0) -> float:
        base = {
            'TREND': 0.62,
            'RANGE': 0.57,
            'HIGH_VOL': 0.54,
            'LOW_VOL': 0.56,
        }.get(mapped_regime, 0.56)
        conf = float(np.clip(detector_confidence, 0.0, 1.0))
        stability = float(np.clip(stability_score, 0.0, 1.0))
        adjusted = base * 0.6 + conf * 0.3 + stability * 0.1
        return float(np.clip(adjusted, 0.50, 0.95))

    def _validate_market_quality_contract(self, market_quality: MarketQualityContract) -> None:
        if market_quality is None:
            raise ValueError('market_quality is required')

        try:
            operable = market_quality.operable
            reason = market_quality.reason
            spread_bps = market_quality.spread_bps
            avg_volume = market_quality.avg_volume
        except AttributeError as exc:
            raise ValueError(f'market_quality missing required attribute: {exc}') from exc

        if not isinstance(operable, bool):
            raise ValueError('market_quality.operable must be bool')
        if not isinstance(reason, str) or not reason:
            raise ValueError('market_quality.reason must be non-empty str')
        if not np.isfinite(float(spread_bps)):
            raise ValueError('market_quality.spread_bps must be finite')
        if not np.isfinite(float(avg_volume)):
            raise ValueError('market_quality.avg_volume must be finite')

    def _relative_liquidity_from_quality(self, market_quality: MarketQualityContract) -> float:
        avg_volume = float(market_quality.avg_volume)
        if not np.isfinite(avg_volume) or avg_volume < 0.0:
            logger.warning('market_quality_avg_volume_invalid avg_volume=%s; using liquidity stress floor', avg_volume)
            return 0.01
        baseline = max(float(self.s.market_min_avg_volume), 1e-9)
        return float(np.clip(avg_volume / baseline, 0.01, 5.0))

    def _cooldown_seconds(self) -> float:
        timeframe_seconds = float(self._timeframe_to_seconds(self.s.timeframe))
        return max(float(self.MIN_SECONDS_BETWEEN_TRADES), timeframe_seconds)

    def _conservative_friction_cost(self, spread_bps: float, volatility: float) -> float:
        base_friction = (2.0 * self.s.taker_fee) + (spread_bps / 10_000.0) + (self.s.slippage_bps / 10_000.0)
        spread_penalty = max(spread_bps / 10_000.0, 0.0) * 0.5
        volatility_penalty = min(max(volatility / max(self.s.volatility_target, 1e-9), 0.0), 3.0) * 0.0006
        return float((base_friction * self.s.friction_safety_multiplier) + spread_penalty + volatility_penalty)

    def _minimum_operational_edge(self, *, effective_min_edge: float, friction_cost: float, regime: str) -> float:
        _ = friction_cost
        _ = regime
        return float(max(effective_min_edge, self.s.operational_edge_floor))

    def _compute_total_equity(self) -> float:
        # Exchange equity is the accounting source of truth for total account equity.
        return float(max(self.state.exchange_equity, 0.0))

    def _repair_state_consistency(self, *, reference_price: float = 0.0) -> None:
        if not np.isfinite(float(self.state.equity)) or self.state.equity < 0.0:
            self.state.equity = max(float(self.state.equity) if np.isfinite(float(self.state.equity)) else 0.0, 0.0)
        if not np.isfinite(float(self.state.exchange_equity)) or self.state.exchange_equity < 0.0:
            self.state.exchange_equity = max(float(self.state.exchange_equity) if np.isfinite(float(self.state.exchange_equity)) else 0.0, 0.0)
        if not np.isfinite(float(self.state.position_qty)) or self.state.position_qty < 0.0:
            self.state.position_qty = 0.0

        if self.state.position_state == PositionState.LONG and self.state.position_qty <= 0.0:
            self.state.position_state = PositionState.FLAT
        if self.state.position_state == PositionState.FLAT and self.state.position_qty > 0.0:
            self.state.position_state = PositionState.LONG

        if self.state.position_state == PositionState.LONG and self.state.entry_price <= 0.0:
            if reference_price > 0.0 and np.isfinite(reference_price):
                self.state.entry_price = float(reference_price)
            else:
                self.state.position_state = PositionState.FLAT
                self.state.position_qty = 0.0
                self.state.entry_price = 0.0
                self.state.position_opened_at = None
                self.state.tp_price = 0.0
                self.state.sl_price = 0.0
                self.state.unrealized_pnl = 0.0
                self.state.capital_in_position = 0.0
                self.state.partial_exits_done.clear()

    def _sync_daily_anchor(self, total_equity: float) -> None:
        today = datetime.now(timezone.utc).date()
        anchor_date = getattr(self, '_daily_anchor_date', today)
        anchor_equity = float(getattr(self, '_daily_anchor_equity', 0.0))
        if today != anchor_date:
            anchor_date = today
            anchor_equity = total_equity
        if anchor_equity <= 0.0:
            anchor_equity = total_equity
        self._daily_anchor_date = anchor_date
        self._daily_anchor_equity = anchor_equity

    def _validate_financial_invariants(self) -> tuple[bool, str]:
        equity = float(max(self.state.exchange_equity, 0.0))
        if self.state.position_qty < -1e-12:
            return False, 'position_qty_negative'
        if self.state.position_state == PositionState.LONG and self.state.position_qty > 0.0 and self.state.entry_price <= 0.0:
            return False, 'entry_price_missing_for_open_position'
        total_pnl = float(self.state.realized_pnl + self.state.unrealized_pnl)
        if equity > 0.0 and abs(total_pnl) > equity:
            return False, 'pnl_exceeds_equity'
        return True, 'ok'

    def _check_kill_switch_state(self) -> tuple[bool, str]:
        if bool(getattr(self.s, 'disable_trading', False)):
            return True, 'disable_trading_env'
        if self.shutdown_event.is_set():
            return True, 'shutdown_requested'
        if self.state.kill_switch:
            return True, self.state.last_kill_switch_trigger or 'kill_switch_active'
        if self.state.rejection_count >= self.s.kill_switch_max_rejections:
            return True, 'too_many_rejections'
        if self.state.avg_latency_ms >= self.s.kill_switch_max_latency_ms:
            return True, 'latency_exceeded'
        if self.initial_equity > 0:
            total_equity = self._compute_total_equity()
            if not np.isfinite(total_equity):
                return True, 'equity_inconsistent'
            drawdown = 1.0 - (total_equity / max(self.initial_equity, 1e-9))
            if drawdown >= self.s.max_global_drawdown:
                return True, 'max_drawdown'
            today = datetime.now(timezone.utc).date()
            daily_anchor_equity = total_equity if today != self._daily_anchor_date else self._daily_anchor_equity
            if daily_anchor_equity <= 0.0:
                daily_anchor_equity = total_equity
            daily_loss_ratio = (daily_anchor_equity - total_equity) / max(daily_anchor_equity, 1e-9)
            if daily_loss_ratio >= self.s.max_daily_loss:
                return True, 'max_daily_loss'
        return False, 'none'

    def _activate_kill_switch_if_needed(self) -> tuple[bool, str]:
        if self.initial_equity > 0:
            self._sync_daily_anchor(self._compute_total_equity())
        blocked, reason = self._check_kill_switch_state()
        if blocked and reason not in {'none', 'shutdown_requested'}:
            self.state.kill_switch = True
            if self.state.last_kill_switch_trigger is None:
                self.state.last_kill_switch_trigger = reason
        return blocked, reason

    def _is_warmup_complete(self, now_ts: float) -> tuple[bool, str]:
        if self.learning_started_at_ms is None:
            return False, 'warmup_missing_start'

        elapsed = max(now_ts - (float(self.learning_started_at_ms) / 1000.0), 0.0)
        bars = int(len(self.data_buffer.ohlcv))
        timeframe = getattr(getattr(self, 's', None), 'timeframe', '5m')
        timeframe_seconds = self._timeframe_to_seconds(timeframe)
        window_seconds = float(getattr(self.data_buffer, 'window_seconds', self.MIN_WARMUP_BARS * timeframe_seconds))
        max_bars_from_window = max(int(window_seconds / timeframe_seconds), 1)
        required_bars = max(20, min(self.MIN_WARMUP_BARS, max_bars_from_window))
        if elapsed < self.MIN_WARMUP_SECONDS or bars < required_bars:
            return False, f'warmup_active elapsed={elapsed:.1f}s bars={bars}/{required_bars}'
        return True, 'warmup_complete'

    def should_block_trading(self) -> bool:
        blocked, reason = self._activate_kill_switch_if_needed()
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
        context = getattr(self, '_context', {}) or {}
        execution_context = {
            'expected_edge': float(context.get('expected_edge') or 0.0),
            'friction_cost': float(context.get('friction_cost') or 0.0),
            'expected_edge_net': float((context.get('expected_edge') or 0.0) - (context.get('friction_cost') or 0.0)),
            'risk_fraction_final': float(context.get('risk_fraction') or getattr(self.s, 'risk_per_trade', 0.0)),
            'portfolio_weight': float(context.get('portfolio_weight') or 1.0),
            'drift_score': float(context.get('drift_score') or 0.0),
            'decision_id': str(context.get('decision_id') or ''),
        }
        try:
            fill = await self.execution_engine.execute(side, qty, execution_context=execution_context)
        except TypeError:
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
        epsilon = 1e-6
        volatility_safe = float(volatility)
        if not np.isfinite(volatility_safe) or volatility_safe <= 0.0:
            volatility_safe = epsilon
        variance = float(max(volatility_safe * volatility_safe, 1e-8))
        conservative_fraction = float(np.clip(self.s.risk_per_trade, 0.001, self.s.max_confidence_allocation))
        expected_edge_safe = float(expected_edge)
        if not np.isfinite(expected_edge_safe):
            return 0.0
        self._edge_scale_samples.append(abs(expected_edge_safe))
        edge_scale_reference = float(np.mean(self._edge_scale_samples)) if self._edge_scale_samples else 0.10
        if not np.isfinite(edge_scale_reference) or edge_scale_reference <= 0.0:
            edge_scale_reference = 0.10

        kelly_fraction = float(max(expected_edge_safe, 0.0) / variance)
        effective_kelly = float(np.clip(kelly_fraction, 0.0, self.s.max_confidence_allocation))
        capped_base = float(min(conservative_fraction, effective_kelly))

        edge_factor = float(np.clip(abs(expected_edge_safe) / max(edge_scale_reference, epsilon), 0.10, 1.0))
        volatility_factor = float(np.clip(self.s.volatility_target / max(volatility_safe, epsilon), 0.25, 1.0))
        drawdown_factor = float(np.clip(1.0 - self._rolling_stats['drawdown_pressure'], 0.5, 1.0))
        ruin_prob = float(np.clip(getattr(self._latest_ruin_snapshot, 'risk_of_ruin_probability', 1.0), 0.0, 1.0))
        ruin_factor = float(np.clip(1.0 - ruin_prob, 0.5, 1.0))
        drift_signal = getattr(self, '_latest_drift_signal', None)
        drift_score = float(getattr(drift_signal, 'drift_score', 0.0)) if drift_signal is not None else 0.0
        if not np.isfinite(drift_score):
            drift_score = 0.0
        drift_factor = float(np.clip(1.0 - drift_score, 0.5, 1.0))

        components = [volatility_safe, drift_score, ruin_prob, edge_factor, volatility_factor, drawdown_factor]
        if not all(np.isfinite(c) for c in components):
            return 0.0

        risk_fraction = capped_base * edge_factor * volatility_factor * drawdown_factor * ruin_factor * drift_factor
        if not np.isfinite(risk_fraction):
            return 0.0
        return float(np.clip(risk_fraction, 0.0, self.s.max_confidence_allocation))

    def _refresh_rolling_stats(self) -> None:
        if not self._trade_pnls:
            return
        arr = np.asarray(self._trade_pnls, dtype=float)
        wins = arr[arr > 0.0]
        losses = arr[arr <= 0.0]
        self._rolling_stats['expectancy'] = float(arr.mean())
        self._rolling_stats['winrate'] = float(wins.size / max(arr.size, 1))
        gross_loss = abs(float(losses.sum()))
        self._rolling_stats['profit_factor'] = float(wins.sum() / max(gross_loss, 1e-9)) if wins.size else 0.0
        std = float(arr.std() or 0.0)
        self._rolling_stats['sharpe'] = float((arr.mean() / std) * np.sqrt(min(arr.size, 252))) if std > 0 else 0.0
        running = np.cumsum(arr)
        peak = np.maximum.accumulate(running)
        drawdowns = peak - running
        self._rolling_stats['drawdown_pressure'] = float(np.clip((drawdowns[-1] / max(np.max(np.abs(peak)), 1e-9)) if drawdowns.size else 0.0, 0.0, 1.0))

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

        edge_modifier = float(np.clip((1.1 - (0.25 * vol_ratio)) * timeframe_factor, 0.35, 1.4))
        confidence_modifier = float(np.clip(0.85 + (0.20 * vol_ratio) + (0.08 * spread_ratio), 0.75, 1.4))

        base_edge = self.decision_engine.min_edge
        confidence_base = self.s.confidence_hold_threshold
        effective_min_edge = float(max(base_edge * edge_modifier, self.s.minimal_mode_min_edge_floor))
        confidence_threshold = float(np.clip(confidence_base * confidence_modifier, 0.50, 0.95))

        regime_uncertain = False

        return effective_min_edge, confidence_threshold, regime_uncertain

    async def _minimum_order_notional(self, *, last_price: float) -> tuple[float, str]:
        exchange_min_notional = float(await self.execution_engine.get_symbol_min_notional(reference_price=last_price))
        configured_floor = float(max(
            getattr(self.s, 'minimal_fixed_position_notional', 0.0),
            getattr(self.s, 'minimal_economic_notional', 0.0),
        ))
        min_notional = float(max(exchange_min_notional, configured_floor))
        self.state.binance_min_notional = exchange_min_notional

        if min_notional <= 0.0 or not np.isfinite(min_notional):
            return 0.0, 'invalid_min_notional'
        if self.state.equity < min_notional:
            return min_notional, 'insufficient_equity_for_min_notional'
        return min_notional, 'ok'

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

    def _run_adaptive_market_making_engine(self, sig: dict[str, Any], last_price: float) -> StrategyVote:
        volatility = float(max(sig.get('volatility') or 0.0, 0.0))
        atr = float(max(sig.get('atr') or 0.0, 0.0))
        prob_up = float(np.clip(self.decision_engine.last_scores.get('global_probability', 0.5), 0.0, 1.0))
        quote = self.market_making_engine.compute_quotes(
            mid_price=max(float(last_price), 1e-9),
            atr=atr,
            volatility=volatility,
            sigma=volatility,
            time_horizon=1.0,
            vpin=float(np.clip(sig.get('signal_vector', {}).get('noise_penalty', 0.0), 0.0, 1.0)),
            liquidity_shock=bool(self._last_market_quality.gap_ratio > max(float(self.s.max_gap_ratio), 1e-9)),
            transformer_prob_up=prob_up,
        )
        directional_skew = (quote.reservation_price - float(last_price)) / max(float(last_price), 1e-9)
        confidence = float(np.clip(abs(directional_skew) * 400.0, 0.0, 1.0))
        if directional_skew > 0.00025:
            decision = 'BUY'
        elif directional_skew < -0.00025:
            decision = 'SELL'
        else:
            decision = 'HOLD'
        return StrategyVote(
            strategy='adaptive_market_making',
            decision=decision,
            confidence=confidence,
            reason=f"mm_skew={directional_skew:.6f}",
        )

    def _run_multi_exchange_arbitrage_engine(self, sig: dict[str, Any]) -> StrategyVote:
        if not self.arbitrage_engine_enabled:
            return StrategyVote('multi_exchange_arbitrage', 'HOLD', 0.0, 'arbitrage_disabled')

        expected_edge = float(self._context.get('expected_edge', 0.0))
        friction_cost = float(self._context.get('friction_cost', 0.0))
        spread_bps = float(max(self._last_market_quality.spread_bps, 0.0))
        arbitrage_signal = float(expected_edge - friction_cost - (spread_bps / 10_000.0))
        confidence = float(np.clip(abs(arbitrage_signal) * 220.0, 0.0, 1.0))

        if arbitrage_signal > 0.0002:
            decision = 'BUY'
        elif arbitrage_signal < -0.0002:
            decision = 'SELL'
        else:
            decision = 'HOLD'

        return StrategyVote(
            strategy='multi_exchange_arbitrage',
            decision=decision,
            confidence=confidence,
            reason=f"arb_edge={arbitrage_signal:.6f}",
        )

    def _route_strategies(self, directional_decision: str, sig: dict[str, Any], last_price: float) -> tuple[str, list[str]]:
        enabled = set(getattr(self.s, 'enabled_strategies', ()))
        reasons: list[str] = []
        votes: list[StrategyVote] = []

        if 'directional' in enabled:
            directional_confidence = float(np.clip(self.decision_engine.last_confidence, 0.0, 1.0))
            votes.append(StrategyVote('directional', directional_decision, directional_confidence, self.decision_engine.last_reason))
        else:
            reasons.append('directional_disabled')

        if 'adaptive_market_making' in enabled:
            mm_vote = self._run_adaptive_market_making_engine(sig=sig, last_price=last_price)
            votes.append(mm_vote)
            reasons.append(f"adaptive_market_making:{mm_vote.decision}:{mm_vote.confidence:.3f}")

        if 'multi_exchange_arbitrage' in enabled:
            arb_vote = self._run_multi_exchange_arbitrage_engine(sig=sig)
            votes.append(arb_vote)
            reasons.append(f"multi_exchange_arbitrage:{arb_vote.decision}:{arb_vote.confidence:.3f}")

        actionable_votes = [vote for vote in votes if vote.decision in {'BUY', 'SELL'}]
        if not actionable_votes:
            return 'HOLD', reasons

        best_vote = max(actionable_votes, key=lambda vote: vote.confidence)
        top_votes = [vote for vote in actionable_votes if abs(vote.confidence - best_vote.confidence) <= 1e-9]
        top_decisions = {vote.decision for vote in top_votes}
        if len(top_decisions) > 1:
            reasons.append('strategy_conflict_equal_confidence')
            return 'HOLD', reasons

        reasons.append(f"selected_strategy={best_vote.strategy}")
        return best_vote.decision, reasons


    async def _assess_institutional_risk(
        self,
        *,
        decision: str,
        order_qty: float,
        last_price: float,
        sig: dict[str, Any],
        returns_series: pd.Series,
    ) -> tuple[bool, str, float]:
        if decision not in {'BUY', 'SELL'}:
            return True, 'not_applicable', order_qty
        if order_qty <= 0.0:
            return False, 'order_qty_zero', 0.0

        requested_notional = max(order_qty * max(last_price, 0.0), 0.0)
        available_liquidity = max(float(self._last_market_quality.avg_volume) * max(last_price, 0.0), 0.0)
        if requested_notional > 0.0 and available_liquidity > 0.0:
            liquidity_coverage = available_liquidity / max(requested_notional, 1e-9)
            if liquidity_coverage < 1.0:
                return False, 'institutional_liquidity_coverage_low', 0.0

        ticket = self.capital_governor.issue_ticket(
            strategy='directional',
            exchange='binance',
            symbol=self.s.symbol,
            requested_notional=requested_notional,
            pnl_or_returns=list(self._trade_returns),
            spread_bps=float(self._last_market_quality.spread_bps),
            available_liquidity=max(available_liquidity, requested_notional),
            price_gap_pct=float(self._last_market_quality.gap_ratio),
            estimated_trade_risk=max(requested_notional * float(self.s.risk_per_trade), 0.0),
        )

        returns_matrix = pd.DataFrame({self.s.symbol: returns_series.tail(300).astype(float)})
        notional_by_exchange = {'binance': float(self.state.capital_in_position)}
        total_exposure = float(self.state.capital_in_position)
        risk_assessment = self.institutional_risk_manager.assess(
            symbol=self.s.symbol,
            side=decision,
            equity=float(self.state.equity),
            daily_pnl=float(self.state.realized_pnl),
            current_price=float(last_price),
            atr=float(sig.get('atr') or 0.0),
            annualized_volatility=float(sig.get('volatility') or 0.0),
            volatility_multiplier=1.0,
            expected_win_rate=max(float(self.decision_engine.last_confidence), 0.5),
            avg_win=max(float(np.mean([x for x in self._trade_pnls if x > 0.0])) if any(x > 0.0 for x in self._trade_pnls) else 1.0, 1e-9),
            avg_loss=max(abs(float(np.mean([x for x in self._trade_pnls if x <= 0.0])) if any(x <= 0.0 for x in self._trade_pnls) else 1.0), 1e-9),
            returns_matrix=returns_matrix,
            risk_per_trade_override=float(self._context.get('risk_fraction') or self.s.risk_per_trade),
            exchange='binance',
            notional_by_exchange=notional_by_exchange,
            total_exposure=total_exposure,
            capital_ticket=ticket,
        )

        if not risk_assessment.allowed:
            return False, f'institutional_risk:{risk_assessment.reason}', 0.0

        if decision == 'BUY':
            capped_qty = min(order_qty, float(risk_assessment.position_size))
        else:
            capped_qty = min(order_qty, float(max(self.state.position_qty, 0.0)))
        if capped_qty <= 0.0:
            return False, 'institutional_risk:size_capped_to_zero', 0.0
        return True, 'ok', capped_qty

    def _is_within_allowed_session(self, now: datetime) -> bool:
        if not self.s.allowed_sessions_utc:
            return True
        hour = int(now.astimezone(timezone.utc).hour)
        for start, end in self.s.allowed_sessions_utc:
            if int(start) <= hour < int(end):
                return True
        return False

    async def _mtf_decision(self) -> tuple[str, str]:
        try:
            mtf_market_data = MarketDataService(self.client, self.s.symbol, self.s.mtf_timeframe)
            mtf_ohlcv = await mtf_market_data.latest_ohlcv(limit=300)
            if len(mtf_ohlcv) < self.MIN_WARMUP_BARS:
                return 'HOLD', 'mtf_insufficient_data'
            mtf_signal = self.signal_engine.generate(mtf_ohlcv, self._last_market_quality.spread_bps)
            mtf_regime_data = self.regime_detector.predict(mtf_signal['returns'], mtf_signal['prices'])
            mtf_regime_raw = mtf_regime_data.get('regime', 'range')
            mtf_regime = self._map_regime(mtf_regime_raw)
            mtf_regime_prob = self._regime_probability(
                mapped_regime=mtf_regime,
                detector_confidence=float(mtf_regime_data.get('confidence') or 0.5),
                stability_score=float(getattr(self, '_latest_regime_snapshot', {}).get('regime_stability_score', 0.0)),
            )
            mtf_breakdown = self.signal_combiner.combine(
                mtf_signal['model_scores']['momentum'],
                mtf_signal['model_scores']['mean_reversion'],
                mtf_regime_prob,
                mtf_regime,
            )
            mtf_probability = float(mtf_breakdown.combined)
            mtf_edge = float(mtf_probability - 0.5)
            if abs(mtf_signal['skew']) > 2.0 or mtf_signal['kurtosis'] > 8.0:
                mtf_edge *= 0.5
            mtf_friction_cost = float((2.0 * self.s.taker_fee) + (self._last_market_quality.spread_bps / 10_000.0) + (self.s.slippage_bps / 10_000.0))
            mtf_effective_min_edge, mtf_confidence_threshold, mtf_regime_uncertain = self._dynamic_thresholds(
                volatility=float(mtf_signal['volatility']),
                spread_bps=self._last_market_quality.spread_bps,
                regime=mtf_regime,
            )
            self.decision_engine.update_context(
                momentum=float(mtf_breakdown.momentum),
                reversion=float(mtf_breakdown.mean_reversion),
                global_probability=mtf_probability,
                expected_edge=mtf_edge,
                friction_cost=mtf_friction_cost,
                trading_enabled=not self.state.kill_switch,
                market_operable=True,
                effective_min_edge=mtf_effective_min_edge,
                confidence_threshold=mtf_confidence_threshold,
                regime_uncertain=mtf_regime_uncertain,
                reason_prefix='phase=mtf;',
            )
            return self.decision_engine.decide(), 'ok'
        except Exception:
            return 'HOLD', 'mtf_insufficient_data'

    def _volatility_size_multiplier(self, volatility: float) -> float:
        if not self.s.enable_volatility_sizing:
            return 1.0
        safe_vol = max(float(volatility), 1e-9)
        target = max(float(self.s.vol_target_risk), 1e-9)
        raw = target / safe_vol
        return float(np.clip(raw, self.s.vol_min_multiplier, self.s.vol_max_multiplier))

    async def _process_partial_exit(self, now: datetime, last_price: float, volatility: float) -> bool:
        if not self.s.enable_partial_exits:
            return False
        if self.state.position_state != PositionState.LONG or self.state.position_qty <= 0.0:
            return False
        gross_pnl = (last_price - self.state.entry_price) * self.state.position_qty
        estimated_exit_fee = (self.state.position_qty * last_price) * float(self.s.taker_fee)
        if gross_pnl <= estimated_exit_fee:
            return False

        atr_proxy = max(float(volatility), 1e-9)
        for idx, level in enumerate(self.s.partial_exit_levels):
            if idx in self.state.partial_exits_done:
                continue
            trigger_price = self.state.entry_price + (atr_proxy * float(level))
            if last_price < trigger_price:
                continue

            fraction = float(self.s.partial_exit_fractions[idx])
            partial_qty = self.state.position_qty * fraction
            if partial_qty <= 0.0:
                self.state.partial_exits_done.add(idx)
                continue

            fill = await self._execute_order('SELL', partial_qty)
            if not fill:
                return False

            fill_price = float(fill['price'] or last_price)
            filled_qty = float(fill['qty'])
            realized_gross = (fill_price - self.state.entry_price) * filled_qty
            close_fee = (filled_qty * fill_price) * float(self.s.taker_fee)
            net_pnl = realized_gross - close_fee
            self.state.fees_paid += close_fee
            self.state.realized_pnl += net_pnl
            self.execution_engine.register_realized_pnl(net_pnl)
            self.state.position_qty = max(self.state.position_qty - filled_qty, 0.0)
            self.state.partial_exits_done.add(idx)
            self.state.last_block_reason = 'none'
            self.system_state = SystemState.IN_POSITION.value
            self.activity_text = f'Partial exit ejecutado nivel={level:.2f} fracción={fraction:.2f}'

            if self.state.position_qty <= 0.0:
                self.state.position_state = PositionState.FLAT
                self.state.entry_price = 0.0
                self.state.position_opened_at = None
                self.state.tp_price = 0.0
                self.state.sl_price = 0.0
                self.state.unrealized_pnl = 0.0
                self.state.capital_in_position = 0.0
                self.state.partial_exits_done.clear()
                self.system_state = SystemState.COOLDOWN.value
            return True
        return False

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
        critical_error: str = '',
    ) -> None:
        total_equity = self._compute_total_equity()
        capital_actual = max(float(self.state.exchange_equity), 0.0)
        self.peak_equity = max(float(getattr(self, 'peak_equity', 0.0)), total_equity)
        peak_drawdown = max(0.0, self.peak_equity - total_equity)
        now = datetime.now(timezone.utc)
        position_time = (now - self.state.position_opened_at).total_seconds() if self.state.position_opened_at else 0.0
        cooldown = 0.0
        last_trade_ts = getattr(self, "last_trade_ts", None)
        if last_trade_ts:
            cooldown = max(self._cooldown_seconds() - (now.timestamp() - float(last_trade_ts)), 0.0)

        if hasattr(self, '_check_kill_switch_state') and hasattr(self, 's'):
            risk_blocked, risk_reason = self._check_kill_switch_state()
        else:
            risk_blocked, risk_reason = False, 'none'
        risk_state = 'BLOCKED' if risk_blocked or self.system_state == SystemState.BLOCKED_BY_RISK.value else 'OK'
        current_regime_perf = self.conditional_performance.summary(getattr(self, '_current_extended_regime', 'UNKNOWN')) if hasattr(self, 'conditional_performance') else {'expectancy': 0.0}
        self.dashboard.update(
            VisualSnapshot(
                price=max(last_price, 0.0),
                equity=capital_actual,
                pnl=self.state.realized_pnl + self.state.unrealized_pnl,
                daily_pnl=total_equity - float(getattr(self, '_daily_anchor_equity', total_equity)),
                drawdown=peak_drawdown,
                edge=float(expected_edge),
                edge_confidence_score=float(getattr(self._latest_edge_snapshot, 'edge_confidence_score', 0.5)),
                edge_t_stat=float(getattr(self._latest_edge_snapshot, 't_stat', 0.0)),
                edge_bayesian_prob=float(getattr(self._latest_edge_snapshot, 'bayesian_prob_edge_positive', 0.5)),
                edge_sprt_state=str(getattr(self._latest_edge_snapshot, 'sprt_state', 'INCONCLUSIVE')),
                risk_of_ruin_probability=float(getattr(self._latest_ruin_snapshot, 'risk_of_ruin_probability', 1.0)),
                regime_stability_score=float(getattr(self, '_latest_regime_snapshot', {}).get('regime_stability_score', 0.0)),
                expectancy=float(getattr(self, '_rolling_stats', {}).get('expectancy', 0.0)),
                regime_expectancy=float(current_regime_perf.get('expectancy', 0.0)),
                volatility=float(abs(getattr(self, '_signal_quality', {}).get('volatility_adjusted_edge', 0.0))),
                system_state=self.system_state,
                decision=decision,
                confidence=getattr(getattr(self, 'decision_engine', None), 'last_confidence', 0.0),
                scores={
                    **dict(getattr(getattr(self, 'decision_engine', None), 'last_scores', {})),
                    'session_allowed': 1.0 if (self._is_within_allowed_session(now) if hasattr(self, 's') else True) else 0.0,
                    'mtf_available': 0.0 if self.state.last_block_reason == 'mtf_insufficient_data' else 1.0,
                    'mtf_conflict': 1.0 if self.state.last_block_reason == 'mtf_conflict' else 0.0,
                    'risk_state_blocked': 1.0 if risk_state == 'BLOCKED' else 0.0,
                    'binance_min_notional': float(self.state.binance_min_notional),
                    'final_order_notional': float(self.state.final_order_notional),
                    'last_kill_switch_trigger': str(self.state.last_kill_switch_trigger or 'none'),
                    'kill_switch_reason': str(risk_reason),
                },
                model_diagnostics={
                    'momentum': {
                        'weight': float(getattr(getattr(self, 'decision_engine', None), 'last_scores', {}).get('momentum_weight', 0.5)),
                        'stability': float(getattr(self, '_signal_quality', {}).get('stability_weight', 0.0)),
                        'ev_net': float(getattr(self, '_signal_quality', {}).get('expected_value_net_costs', 0.0)),
                    },
                    'mean_reversion': {
                        'weight': float(getattr(getattr(self, 'decision_engine', None), 'last_scores', {}).get('reversion_weight', 0.5)),
                        'stability': float(getattr(self, '_signal_quality', {}).get('stability_weight', 0.0)),
                        'ev_net': float(getattr(self, '_signal_quality', {}).get('expected_value_net_costs', 0.0)),
                    },
                },
                regime=regime,
                risk_state=risk_state,
                execution_state=self.execution_status,
                reason=self.state.last_block_reason if self.state.last_block_reason != 'none' else getattr(getattr(self, 'decision_engine', None), 'last_reason', ''),
                critical_error=critical_error,
            )
        )
        db = getattr(self, 'db', None)
        if db is not None and hasattr(db, 'persist_financial_snapshot'):
            payload = {
                'ts': int(now.timestamp() * 1000),
                'symbol': self.s.symbol,
                'equity': float(capital_actual),
                'total_equity': float(total_equity),
                'pnl_total': float(self.state.realized_pnl + self.state.unrealized_pnl),
                'daily_pnl': float(total_equity - float(getattr(self, '_daily_anchor_equity', total_equity))),
                'drawdown_abs': float(peak_drawdown),
                'decision': str(decision),
                'regime': str(regime),
                'system_state': str(self.system_state),
                'risk_state': str(risk_state),
                'reason': str(self.state.last_block_reason),
            }
            with suppress(Exception):
                asyncio.create_task(db.persist_financial_snapshot(payload))

    def _handle_cycle_exception(self, exc: Exception) -> bool:
        self.state.consecutive_cycle_errors += 1
        self.activity_text = f'Error de ciclo: {exc}'
        self.execution_status = 'ERROR'
        self.system_state = SystemState.ERROR.value
        monitoring = getattr(self, 'monitoring', None)
        if monitoring is not None and hasattr(monitoring, 'set_system_degraded'):
            with suppress(Exception):
                monitoring.set_system_degraded(exc)
        if hasattr(self, 'decision_engine'):
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
        self._publish_dashboard('HOLD', 0.0, 0.5, 0.5, 0.5, 'ERROR', 0.0, 'ERROR', critical_error=str(exc))
        if self.state.consecutive_cycle_errors >= self.MAX_CONSECUTIVE_CYCLE_ERRORS:
            self._shutdown_reason = 'max_consecutive_cycle_errors'
            self.shutdown_event.set()
            return True
        return False

    async def run(self) -> None:
        try:
            await self.initialize()
            await self._startup_reconciliation()
            safety_seconds = float(getattr(self.s, 'startup_safety_seconds', 10.0))
            self._startup_safety_until_monotonic = time.monotonic() + max(safety_seconds, 0.0)
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
                                    if time.monotonic() < self._startup_safety_until_monotonic:
                                        self.system_state = SystemState.LEARNING_MARKET.value
                                        self.state.last_block_reason = 'startup_reconciliation_safety_mode'
                                        self.activity_text = 'Startup safety mode: reconciliation barrier active'
                                        self._publish_dashboard('HOLD', 0.0, 0.5, 0.5, 0.5, 'RANGE', float(ohlcv['close'].iloc[-1]), 'OK')
                                        await asyncio.sleep(self.s.loop_interval_seconds)
                                        continue
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
                                    self._validate_market_quality_contract(self._last_market_quality)

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
                                    ok_balance, free_usdt, exchange_equity, _, reason = await self._fetch_account_balance()
                                    if not ok_balance:
                                        self.system_state = SystemState.BLOCKED_BY_RISK.value
                                        self.state.last_block_reason = reason
                                        self.activity_text = f'Balance inválido: {reason}'
                                        self._publish_dashboard('DISABLE_TRADING', 0.0, 0.5, 0.5, 0.5, 'RANGE', float(ohlcv["close"].iloc[-1]), 'ERROR')
                                        await asyncio.sleep(self.s.loop_interval_seconds)
                                        continue

                                    self.state.equity = free_usdt
                                    self.state.exchange_equity = exchange_equity
                                    total_eq = max(float(self.state.exchange_equity), 1e-9)
                                    self.execution_engine.update_firewall_limits(
                                        max_total_exposure=total_eq * float(self.s.max_total_exposure),
                                        max_asset_exposure=total_eq * float(self.s.max_asset_exposure),
                                        max_daily_loss=total_eq * float(self.s.max_daily_loss),
                                        max_daily_notional=max(total_eq * 4.0, total_eq),
                                    )
                                    self.capital_governor.update_state(
                                        strategy='directional',
                                        exchange='binance',
                                        symbol=self.s.symbol,
                                        capital_by_strategy=float(self.state.capital_in_position),
                                        capital_by_exchange=float(self.state.capital_in_position),
                                        total_exposure=float(self.state.capital_in_position),
                                        asset_exposure=float(self.state.capital_in_position),
                                        exchange_equity=float(self.state.exchange_equity),
                                        daily_pnl=float(self._compute_total_equity() - float(getattr(self, '_daily_anchor_equity', self._compute_total_equity()))),
                                    )
                                    sig = await asyncio.to_thread(self.signal_engine.generate, ohlcv, spread_bps)
                                    last_price = float(sig['prices'].iloc[-1])

                                    if self.state.position_state == PositionState.LONG:
                                        self.state.unrealized_pnl = (last_price - self.state.entry_price) * self.state.position_qty
                                        self.state.capital_in_position = self.state.position_qty * last_price
                                    else:
                                        self.state.unrealized_pnl = 0.0
                                        self.state.capital_in_position = 0.0

                                    self._repair_state_consistency(reference_price=last_price)
                                    valid_financials, invariant_reason = self._validate_financial_invariants()
                                    if not valid_financials:
                                        self.state.kill_switch = True
                                        self.state.last_kill_switch_trigger = invariant_reason
                                        self.system_state = SystemState.BLOCKED_BY_RISK.value
                                        self.state.last_block_reason = invariant_reason
                                        self.activity_text = f'Financial invariant violation: {invariant_reason}'
                                        self._publish_dashboard('DISABLE_TRADING', 0.0, 0.5, 0.5, 0.5, 'RANGE', float(ohlcv["close"].iloc[-1]), 'ERROR')
                                        await asyncio.sleep(self.s.loop_interval_seconds)
                                        continue

                                    regime_data = self.regime_detector.predict(sig['returns'], sig['prices'])
                                    regime_raw = regime_data.get('regime', 'range')
                                    regime = self._map_regime(regime_raw)
                                    returns_series = pd.Series(sig['returns'], dtype=float)
                                    autocorr = float(returns_series.tail(80).autocorr(lag=1) or 0.0)
                                    rel_liquidity = self._relative_liquidity_from_quality(self._last_market_quality)
                                    regime_snapshot = self.regime_controller.update(
                                        volatility=float(sig['volatility']),
                                        autocorr=autocorr,
                                        avg_spread_bps=float(self._last_market_quality.spread_bps),
                                        relative_liquidity=rel_liquidity,
                                    )
                                    self._latest_regime_snapshot = {
                                        'current_regime': regime_snapshot.current_regime,
                                        'regime_stability_score': regime_snapshot.regime_stability_score,
                                    }
                                    self._current_extended_regime = regime_snapshot.current_regime

                                    regime_prob = self._regime_probability(
                                        mapped_regime=regime,
                                        detector_confidence=float(regime_data.get('confidence') or 0.5),
                                        stability_score=float(regime_snapshot.regime_stability_score),
                                    )
                                    breakdown = self.signal_combiner.combine(
                                        sig['model_scores']['momentum'],
                                        sig['model_scores']['mean_reversion'],
                                        regime_prob,
                                        regime,
                                    )

                                    signal_vector = dict(sig.get('signal_vector') or {})
                                    confidence_score = float(np.clip(signal_vector.get('confidence_score', 0.5), 0.0, 1.0))
                                    stability_weight = float(np.clip(signal_vector.get('stability_weight', 0.5), 0.0, 1.0))
                                    regime_alignment = float(np.clip(signal_vector.get('regime_alignment', 0.5), 0.0, 1.0))
                                    noise_penalty = float(np.clip(signal_vector.get('noise_penalty', 0.0), 0.0, 0.7))
                                    corr_penalty = float(np.clip(signal_vector.get('correlation_penalty', 0.0), 0.0, 0.95))
                                    vol_adjusted_edge = float(signal_vector.get('volatility_adjusted_edge', 0.0))

                                    momentum_weight = float(np.clip((0.55 if regime == 'TREND' else 0.40) * stability_weight * (1.0 - (corr_penalty * 0.35)), 0.15, 0.75))
                                    reversion_weight = float(np.clip((0.55 if regime == 'RANGE' else 0.40) * (1.0 - noise_penalty), 0.15, 0.75))
                                    total_weight = max(momentum_weight + reversion_weight, 1e-9)
                                    momentum_weight /= total_weight
                                    reversion_weight /= total_weight

                                    probability = float(np.clip((momentum_weight * breakdown.momentum) + (reversion_weight * breakdown.mean_reversion), 0.0, 1.0))
                                    raw_edge = float(probability - 0.5)
                                    expected_edge = raw_edge * (0.55 + (0.45 * regime_alignment))
                                    expected_edge *= float(np.clip(1.0 - noise_penalty - (0.25 * corr_penalty), 0.25, 1.0))
                                    expected_edge += 0.15 * vol_adjusted_edge

                                    # ajuste por colas pesadas / chop
                                    if abs(sig['skew']) > 2.0 or sig['kurtosis'] > 8.0:
                                        expected_edge *= 0.5
                                    range_compression = (float(sig['prices'].tail(30).max()) - float(sig['prices'].tail(30).min())) / max(last_price, 1e-9)
                                    if range_compression < 0.0015:
                                        expected_edge *= 0.6

                                    if expected_edge < 0.0:
                                        self.state.negative_edge_streak += 1
                                    else:
                                        self.state.negative_edge_streak = 0

                                    friction_cost = self._conservative_friction_cost(self._last_market_quality.spread_bps, float(sig['volatility']))
                                    self._asset_price_history[self.s.symbol] = pd.Series(sig['prices']).astype(float).tail(300)
                                    self._asset_edges[self.s.symbol] = float(expected_edge)
                                    self._asset_volatility[self.s.symbol] = float(max(sig['volatility'], 1e-6))
                                    corr_matrix = self.exposure_model.compute_correlation_matrix(self._asset_price_history)
                                    self._portfolio_weights = self.portfolio_allocator.allocate(
                                        asset_edges=self._asset_edges,
                                        asset_volatility=self._asset_volatility,
                                        correlation_matrix=corr_matrix,
                                    )

                                    returns_series = pd.Series(sig['returns']).astype(float).dropna()
                                    recent_returns = returns_series.tail(60).to_numpy(dtype=float)
                                    historical_returns = returns_series.tail(240).to_numpy(dtype=float)
                                    cusum_score = self.cusum_drift.update(float(expected_edge))
                                    kl_raw = self.kl_drift.compute(recent_returns, historical_returns)
                                    kl_score = float(np.clip(kl_raw / 0.5, 0.0, 1.0))
                                    if not np.isfinite(cusum_score):
                                        cusum_score = 0.0
                                    if not np.isfinite(kl_score):
                                        kl_score = 0.0
                                    drift_score = float(np.clip(0.5 * cusum_score + 0.5 * kl_score, 0.0, 1.0))
                                    self._latest_drift_signal = DriftSignal(
                                        drift_score=drift_score,
                                        regime_change_probability=drift_score,
                                    )

                                    expected_value_net_costs = float(expected_edge - friction_cost)
                                    self._signal_quality.update(
                                        {
                                            'confidence_score': confidence_score,
                                            'stability_weight': stability_weight,
                                            'regime_alignment': regime_alignment,
                                            'volatility_adjusted_edge': vol_adjusted_edge,
                                            'expected_value_net_costs': expected_value_net_costs,
                                        }
                                    )

                                    effective_min_edge, confidence_threshold, regime_uncertain = self._dynamic_thresholds(
                                        volatility=float(sig['volatility']),
                                        spread_bps=self._last_market_quality.spread_bps,
                                        regime=regime,
                                    )
                                    effective_min_edge = self.frequency_controller.adjust_threshold(
                                        dynamic_edge_threshold=effective_min_edge,
                                        friction_cost=friction_cost,
                                        now=now,
                                    )
                                    self.decision_engine.last_scores['volatility'] = float(sig['volatility'])
                                    self.decision_engine.last_scores['momentum_weight'] = momentum_weight
                                    self.decision_engine.last_scores['reversion_weight'] = reversion_weight
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
                                    self._context = dict(getattr(self.decision_engine, '_context', {}) or {})
                                    self._context.setdefault('risk_fraction', float(self.s.risk_per_trade))
                                    self._context.setdefault('portfolio_weight', 1.0)
                                    self._context.setdefault('drift_score', 0.0)
                                    statistical_decision = self.decision_engine.decide()
                                    routed_decision, routing_reasons = self._route_strategies(statistical_decision, sig, last_price)
                                    self.decision_engine.last_scores['volatility'] = float(sig['volatility'])
                                    self.decision_engine.last_scores['momentum_weight'] = momentum_weight
                                    self.decision_engine.last_scores['reversion_weight'] = reversion_weight
                                    executable_decision = routed_decision
                                    policy_block_reasons: list[str] = list(routing_reasons)
                                    self.state.pending_action = routed_decision if routed_decision in {'BUY', 'SELL'} else None
                                    self.state.last_block_reason = self.decision_engine.last_reason
                                    self.activity_text = self.decision_engine.last_reason

                                    if self.state.position_state == PositionState.LONG and (self.state.position_qty <= 0.0 or self.state.entry_price <= 0.0):
                                        executable_decision = 'HOLD'
                                        policy_block_reasons.append('STATE_INCONSISTENT')
                                        self.system_state = SystemState.BLOCKED_BY_RISK.value
                                        self.state.last_block_reason = 'state_inconsistent'
                                        self.activity_text = 'BLOCK por estado inconsistente de posición'
                                    if self.state.negative_edge_streak >= 8 and self.state.position_state == PositionState.FLAT:
                                        executable_decision = 'HOLD'
                                        policy_block_reasons.append('PERSISTENT_NEGATIVE_EDGE')
                                        self.system_state = SystemState.BLOCKED_BY_RISK.value
                                        self.state.last_block_reason = 'persistent_negative_edge'
                                        self.activity_text = 'BLOCK por edge negativo persistente'

                                    cooldown = max(self._cooldown_seconds() - (now.timestamp() - self.last_trade_ts), 0.0) if self.last_trade_ts else 0.0
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

                                    if self.s.enable_session_filter and self.state.position_state == PositionState.FLAT and executable_decision == 'BUY':
                                        if not self._is_within_allowed_session(now):
                                            executable_decision = 'HOLD'
                                            policy_block_reasons.append('SESSION_FILTER')
                                            self.state.last_block_reason = 'session_filter'
                                            self.activity_text = 'HOLD por session_filter'

                                    if self.s.enable_mtf_confirmation and executable_decision in {'BUY', 'SELL'}:
                                        mtf_decision, mtf_reason = await self._mtf_decision()
                                        if mtf_decision == 'HOLD' and mtf_reason == 'mtf_insufficient_data':
                                            executable_decision = 'HOLD'
                                            policy_block_reasons.append('MTF_INSUFFICIENT_DATA')
                                            self.state.last_block_reason = 'mtf_insufficient_data'
                                            self.activity_text = 'HOLD por mtf_insufficient_data'
                                        elif mtf_decision != executable_decision:
                                            if self.s.mtf_confirmation_mode in {'confirm', 'veto'}:
                                                executable_decision = 'HOLD'
                                                policy_block_reasons.append('MTF_CONFLICT')
                                                self.state.last_block_reason = 'mtf_conflict'
                                                self.activity_text = 'HOLD por mtf_conflict'

                                    if executable_decision in {'BUY', 'SELL'}:
                                        protection = await self.capital_protection_controller.evaluate(
                                            symbol=self.s.symbol,
                                            position_qty=self.state.position_qty,
                                        )
                                        if not protection.allowed:
                                            executable_decision = 'HOLD'
                                            policy_block_reasons.append(protection.reason)
                                            self.state.last_block_reason = protection.reason
                                            self.activity_text = f'HOLD por {protection.reason}'

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
                                    partial_exit_done = await self._process_partial_exit(now, last_price, volatility)
                                    if partial_exit_done:
                                        decision = 'HOLD'

                                    variance_now = float(np.var(np.asarray(self._trade_returns, dtype=float))) if self._trade_returns else float(max(volatility * volatility, 1e-8))
                                    preview_fraction = float(np.clip(self.s.risk_per_trade, 0.0, self.s.max_confidence_allocation))
                                    self._latest_ruin_snapshot = self.risk_of_ruin.estimate(
                                        edge=expected_value_net_costs,
                                        variance=variance_now,
                                        position_fraction=preview_fraction,
                                        capital=self.state.equity,
                                    )
                                    risk_fraction = self._risk_fraction(expected_edge=expected_value_net_costs, volatility=volatility)
                                    portfolio_weight = float(self._portfolio_weights.get(self.s.symbol, 1.0))
                                    self._context['risk_fraction'] = float(risk_fraction)
                                    self._context['portfolio_weight'] = float(portfolio_weight)
                                    self._context['drift_score'] = float(getattr(self._latest_drift_signal, 'drift_score', 0.0))
                                    if not np.isfinite(portfolio_weight):
                                        portfolio_weight = 1.0
                                    portfolio_weight = float(max(portfolio_weight, 0.0))
                                    if portfolio_weight > 1.0:
                                        portfolio_weight = 1.0
                                    if len(self._portfolio_weights) <= 1 and portfolio_weight <= 0.0:
                                        portfolio_weight = 1.0
                                    risk_fraction *= portfolio_weight
                                    order_qty = 0.0
                                    self.state.final_order_notional = 0.0
                                    if decision == 'BUY':
                                        min_notional, min_notional_reason = await self._minimum_order_notional(last_price=last_price)
                                        if min_notional_reason == 'insufficient_equity_for_min_notional':
                                            decision = 'HOLD'
                                            self.system_state = SystemState.BLOCKED_BY_RISK.value
                                            self.state.last_block_reason = 'insufficient_equity_for_min_notional'
                                            self.activity_text = 'BLOCK por equity insuficiente para mínimo notional'
                                            policy_block_reasons.append('INSUFFICIENT_EQUITY_FOR_MIN_NOTIONAL')
                                        elif min_notional <= 0.0:
                                            decision = 'HOLD'
                                            self.system_state = SystemState.BLOCKED_BY_RISK.value
                                            self.state.last_block_reason = 'invalid_min_notional'
                                            self.activity_text = 'BLOCK por mínimo notional inválido'
                                            policy_block_reasons.append('INVALID_MIN_NOTIONAL')
                                        elif risk_fraction <= 0:
                                            decision = 'HOLD'
                                            self.system_state = SystemState.WAITING_EDGE.value
                                            self.state.last_block_reason = 'risk_fraction_zero'
                                            self.activity_text = 'HOLD por riesgo insuficiente para sizing'
                                            policy_block_reasons.append('RISK_FRACTION_ZERO')
                                        else:
                                            requested_notional = self.state.equity * risk_fraction
                                            if requested_notional < min_notional:
                                                decision = 'HOLD'
                                                self.system_state = SystemState.BLOCKED_BY_RISK.value
                                                self.state.last_block_reason = 'notional_below_exchange_minimum'
                                                self.activity_text = 'BLOCK por notional por debajo del mínimo del exchange'
                                                policy_block_reasons.append('NOTIONAL_BELOW_EXCHANGE_MINIMUM')
                                            else:
                                                base_qty = requested_notional / max(last_price, 1e-9)
                                                order_qty = base_qty * self._volatility_size_multiplier(volatility)
                                    elif decision == 'SELL':
                                        order_qty = self.state.position_qty

                                    if decision in {'BUY', 'SELL'} and order_qty <= 0.0:
                                        policy_block_reasons.append('ORDER_QTY_ZERO_POLICY')
                                        if self.system_state == SystemState.BLOCKED_BY_RISK.value:
                                            self.state.last_block_reason = self.state.last_block_reason or 'blocked'
                                        decision = 'HOLD'

                                    if decision in {'BUY', 'SELL'} and order_qty > 0.0:
                                        risk_ok, risk_reason, risk_capped_qty = await self._assess_institutional_risk(
                                            decision=decision,
                                            order_qty=order_qty,
                                            last_price=last_price,
                                            sig=sig,
                                            returns_series=returns_series,
                                        )
                                        if not risk_ok:
                                            decision = 'HOLD'
                                            policy_block_reasons.append(risk_reason)
                                            self.system_state = SystemState.BLOCKED_BY_RISK.value
                                            self.state.last_block_reason = risk_reason
                                            self.activity_text = f'BLOCK por {risk_reason}'
                                        else:
                                            order_qty = float(risk_capped_qty)

                                    self._record_decision_audit(
                                        decision=decision,
                                        expected_edge=expected_edge,
                                        effective_min_edge=effective_min_edge,
                                        confidence=getattr(getattr(self, 'decision_engine', None), 'last_confidence', 0.0),
                                        confidence_threshold=confidence_threshold,
                                        friction_cost=friction_cost,
                                        regime_uncertain=regime_uncertain,
                                        policy_block_reasons=policy_block_reasons,
                                    )

                                    decision_id = f"dec-{uuid.uuid4()}"
                                    self._context['decision_id'] = decision_id
                                    await self.db.persist_decision_snapshot(
                                        decision_id,
                                        {
                                            'ts': datetime.now(timezone.utc).isoformat(),
                                            'symbol': self.s.symbol,
                                            'decision': decision,
                                            'expected_edge': float(expected_edge),
                                            'risk_fraction': float(risk_fraction),
                                            'momentum': float(breakdown.momentum),
                                            'mean_reversion': float(breakdown.mean_reversion),
                                            'regime_probability': float(breakdown.regime),
                                            'confidence': float(self.decision_engine.last_confidence),
                                        },
                                    )

                                    if decision in {'BUY', 'SELL'} and order_qty > 0.0:
                                        if decision == 'SELL':
                                            self.state.final_order_notional = float(max(order_qty * last_price, 0.0))
                                        # --- MINIMAL MODE: enforce exchange minimums ---
                                        if decision == 'BUY':
                                            exchange_min_qty = await self.execution_engine.get_exchange_min_size()
                                            if exchange_min_qty is not None and order_qty < exchange_min_qty:
                                                logger.info(
                                                    "Order qty %.10f < exchange_min_qty %.10f, adjusting to minimum",
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
                                            self.frequency_controller.register_trade(now, trade_id=str(fill.get('id') or ''))
                                            self.state.tp_price, self.state.sl_price = self.execution_engine.compute_dynamic_exit_levels(
                                                entry_price=self.state.entry_price,
                                                atr=float(sig['atr']),
                                                side='BUY',
                                            )
                                            self.state.fees_paid += (self.state.position_qty * fill_price) * float(self.s.taker_fee)
                                            self.state.partial_exits_done.clear()
                                            self.system_state = SystemState.IN_POSITION.value
                                            self.state.last_block_reason = 'none'
                                            self.activity_text = f'BUY ejecutado qty={self.state.position_qty:.6f}'
                                        elif fill and decision == 'SELL':
                                            fill_price = float(fill['price'] or last_price)
                                            filled_qty = float(fill.get('qty') or fill.get('amount') or fill.get('filled') or 0.0)
                                            if filled_qty <= 0.0:
                                                self.system_state = SystemState.BLOCKED_BY_RISK.value
                                                self.state.last_block_reason = 'sell_fill_qty_zero'
                                                self.activity_text = 'SELL sin fill qty válido'
                                                continue
                                            close_qty = min(filled_qty, max(self.state.position_qty, 0.0))
                                            gross_pnl = (fill_price - self.state.entry_price) * close_qty
                                            close_fee = (close_qty * fill_price) * float(self.s.taker_fee)
                                            self.state.fees_paid += close_fee
                                            net_pnl = gross_pnl - close_fee
                                            self.state.realized_pnl += net_pnl
                                            self.execution_engine.register_realized_pnl(net_pnl)
                                            self._trade_pnls.append(float(net_pnl))
                                            entry_notional = max(self.state.entry_price * max(close_qty, 1e-9), 1e-9)
                                            realized_return = float(net_pnl / entry_notional)
                                            self._trade_returns.append(realized_return)
                                            self.conditional_performance.record_trade(self._current_extended_regime, float(net_pnl), realized_return)
                                            self._latest_edge_snapshot = self.edge_monitor.update(realized_return)
                                            self._refresh_rolling_stats()
                                            self.state.consecutive_losses = self.state.consecutive_losses + 1 if net_pnl <= 0 else 0
                                            if net_pnl > 0:
                                                self.state.winning_trades += 1
                                            self.state.position_qty = max(self.state.position_qty - close_qty, 0.0)
                                            if self.state.position_qty <= 0.0:
                                                self.state.position_state = PositionState.FLAT
                                                self.state.position_qty = 0.0
                                                self.state.entry_price = 0.0
                                                self.state.position_opened_at = None
                                                self.state.tp_price = 0.0
                                                self.state.sl_price = 0.0
                                                self.state.unrealized_pnl = 0.0
                                                self.state.capital_in_position = 0.0
                                                self.state.partial_exits_done.clear()
                                            self.last_trade_ts = now.timestamp()
                                            self.state.trades += 1
                                            self.frequency_controller.register_trade(now, trade_id=str(fill.get('id') or ''))
                                            self.system_state = SystemState.COOLDOWN.value
                                            self.state.last_block_reason = 'none'
                                        else:
                                            self.system_state = SystemState.BLOCKED_BY_RISK.value

                                    self.state.consecutive_cycle_errors = 0
                                    dashboard_decision = 'BLOCK' if decision == 'HOLD' and self.system_state == SystemState.BLOCKED_BY_RISK.value else decision
                                    self._publish_dashboard(
                                        decision=dashboard_decision,
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
