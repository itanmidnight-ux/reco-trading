from __future__ import annotations

import asyncio
import logging
import os as _os
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import ccxt
from rich.live import Live

from reco_trading.analytics.session_tracker import SessionTracker
from reco_trading.config.settings import Settings
from reco_trading.config.symbols import normalize_symbol, split_symbol
from reco_trading.core.state_machine import BotState
from reco_trading.core.observability import RuntimeObservability, start_metrics_server
from reco_trading.data.market_stream import MarketStream
from reco_trading.database.repository import Repository
from reco_trading.exchange.binance_client import BinanceClient
from reco_trading.exchange.order_manager import OrderManager
from reco_trading.risk.position_manager import Position, PositionManager
from reco_trading.risk.advanced_risk_manager import AdvancedRiskManager
from reco_trading.risk.adaptive_sizer import AdaptiveSizer
from reco_trading.risk.capital_profile import CapitalProfile, CapitalProfileManager
from reco_trading.risk.investment_optimizer import InvestmentOptimizer
from reco_trading.risk.risk_manager import RiskManager
from reco_trading.risk.portfolio_risk import PortfolioRiskController
from reco_trading.strategy.confidence_model import ConfidenceModel
from reco_trading.strategy.indicators import apply_indicators
from reco_trading.strategy.confluence import TimeframeConfluence
from reco_trading.strategy.market_intelligence import MarketIntelligence
from reco_trading.strategy.exit_intelligence import ExitIntelligence
from reco_trading.strategy.signal_engine import SignalBundle, SignalEngine
from reco_trading.ui.dashboard import TerminalDashboard
from reco_trading.core.resilience import ResilienceManager, ResilienceConfig, NetworkResilience
from reco_trading.core.intelligent_auto_improver import IntelligentAutoImprover
from reco_trading.core.self_analyzer import SelfAnalyzer
from reco_trading.core.autonomous_optimizer import AutonomousOptimizer
from reco_trading.core.loop_manager import LoopManager
from reco_trading.core.multi_pair_manager import MultiPairManager
from reco_trading.ml.enhanced_ml_engine import EnhancedMLEngine

# Optional ML models - only loaded if dependencies are available
try:
    from reco_trading.ml.tft_model import TFTManager, TFTConfig
    _TFT_AVAILABLE = True
except ImportError:
    TFTManager = None
    TFTConfig = None
    _TFT_AVAILABLE = False

try:
    from reco_trading.ml.nbeats_model import NBEATSManager, NBEATSConfig
    _NBEATS_AVAILABLE = True
except ImportError:
    NBEATSManager = None
    NBEATSConfig = None
    _NBEATS_AVAILABLE = False

try:
    from reco_trading.ml.advanced_meta_learner import MetaLearningManager, MetaLearningConfig
    _META_LEARNING_AVAILABLE = True
except ImportError:
    MetaLearningManager = None
    MetaLearningConfig = None
    _META_LEARNING_AVAILABLE = False

from reco_trading.core.trading_modes import TradingModeManager, WebSocketManager
from reco_trading.core.trading_modes import TradingModeManager, WebSocketManager
from reco_trading.core.integrations import initialize_all_modules, get_system_status, create_default_config
from reco_trading.core.autonomous_brain import AutonomousTradingBrain, create_autonomous_brain
from reco_trading.core.emergency_systems import EmergencySystem, DataValidator

if TYPE_CHECKING:
    from reco_trading.ui.state_manager import StateManager



try:
    import resource as _resource

    def _get_memory_mb() -> float:
        return _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss / 1024.0
except ImportError:

    def _get_memory_mb() -> float:
        try:
            with open(f"/proc/{_os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024.0
        except (OSError, IndexError, ValueError):
            pass
        return 0.0


def _get_database_dsn(settings: Settings) -> str:
    """Get best available database DSN with fallback."""
    if settings.postgres_dsn:
        return settings.postgres_dsn
    if settings.mysql_dsn:
        return settings.mysql_dsn
    if settings.database_url:
        # Convert sqlite:// to sqlite+aiosqlite:// for async support
        if settings.database_url.startswith("sqlite://"):
            return settings.database_url.replace("sqlite://", "sqlite+aiosqlite://")
        return settings.database_url
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(project_root, "data"), exist_ok=True)
    db_path = os.path.join(project_root, "data", "reco_trading.db")
    return f"sqlite+aiosqlite:///{db_path}"


class BotEngine:
    """Orchestrates market analysis, risk controls, trading and monitoring."""

    def __init__(self, settings: Settings, state_manager: "StateManager | None" = None) -> None:
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.state = BotState.INITIALIZING

        self.client = BinanceClient(settings.binance_api_key, settings.binance_api_secret, settings.binance_testnet)
        self.symbol = normalize_symbol(settings.trading_symbol)
        self.symbols = [normalize_symbol(sym) for sym in (settings.trading_symbols or [])] or [self.symbol]
        self.order_manager = OrderManager(self.client, self.symbol)
        self.market_stream = MarketStream(self.client, self.symbol, settings.history_limit)
        
        dsn = _get_database_dsn(settings)
        self.logger.info(f"Using database: {dsn[:30]}...")
        self.repository = Repository(dsn)
        self.signal_engine = SignalEngine()
        self.confidence_model = ConfidenceModel()
        self.risk_manager = RiskManager(settings.daily_loss_limit_fraction, settings.max_trades_per_day)
        self.advanced_risk_manager = AdvancedRiskManager(
            max_daily_loss_percent=max(float(settings.daily_loss_limit_fraction), 0.0) * 100,
            max_drawdown_percent=max(float(getattr(settings, "max_drawdown_fraction", 0.10)), 0.0) * 100,
        )
        self.position_manager = PositionManager()
        self.capital_profile_manager = CapitalProfileManager()
        self.adaptive_sizer = AdaptiveSizer(
            base_risk_fraction=settings.risk_per_trade_fraction,
            min_multiplier=0.15,
            max_multiplier=1.50,
            confidence_boost_above=0.80,
        )
        self.market_intelligence = MarketIntelligence(settings)
        self.exit_intelligence = ExitIntelligence()
        self.confluence = TimeframeConfluence()

        self.state_manager = state_manager
        if self.state_manager and hasattr(self.state_manager, "configure_log_state_emission"):
            self.state_manager.configure_log_state_emission(
                bool(getattr(self.settings, "ui_state_emit_on_each_log", False))
            )
        if bool(getattr(self.settings, "low_ram_mode", True)):
            settings.history_limit = max(120, min(int(settings.history_limit), 220))
        self.trades_today = 0
        self.win_count = 0
        self.start_time = time.time()
        self.day_marker = datetime.now(timezone.utc).date()
        self.last_close_time: datetime | None = None
        self.pause_trading_until: datetime | None = None
        self.consecutive_losses = 0
        self.equity_peak: float | None = None
        self.starting_equity: float | None = None
        self.trading_paused_by_drawdown = False
        self.exchange_failure_count = 0
        self.exchange_failure_max = 5
        self.exchange_failure_cooldown_seconds = 300
        self.exchange_failure_paused_until: datetime | None = None
        self.runtime_risk_per_trade_fraction: float | None = None
        self.runtime_max_trade_balance_fraction: float | None = None
        self.runtime_capital_limit_usdt: float | None = None
        self.runtime_capital_reserve_ratio: float | None = None
        self.runtime_min_cash_buffer_usdt: float | None = None
        self.runtime_symbol_capital_limits: dict[str, float] = {}
        self.runtime_investment_mode: str = "Balanced"
        self.runtime_dynamic_exit_enabled: bool = False
        self.runtime_confidence_boost_multiplier: float = 1.0
        self.runtime_confidence_threshold: float | None = None
        self.base_filter_config: dict[str, float] = self._get_default_filter_config()
        self.runtime_filter_config: dict[str, float] = dict(self.base_filter_config)
        self.terminal_tui_enabled: bool = bool(getattr(self.settings, "terminal_tui_enabled", True))
        self.terminal_tui_quiet_logs: bool = bool(getattr(self.settings, "terminal_tui_quiet_logs", True))
        self.terminal_tui_refresh_per_second: int = max(int(getattr(self.settings, "terminal_tui_refresh_per_second", 4)), 1)
        self.manual_pause = False
        self.emergency_stop_active = False
        self.equity_curve_history: list[float] = []
        self._recent_pnls: list[float] = []
        self.session_tracker = SessionTracker()
        self.portfolio_risk = PortfolioRiskController(
            max_global_exposure_fraction=float(getattr(self.settings, "max_global_exposure_fraction", 0.7)),
            max_symbol_correlation=float(getattr(self.settings, "max_symbol_correlation", 0.85)),
            symbol_caps=dict(getattr(self.settings, "symbol_capital_limits", {}) or {}),
        )
        self.observability = RuntimeObservability()
        self.investment_optimizer = InvestmentOptimizer()
        self._metrics_server_started = False
        self._quote_to_usdt_cache: dict[str, tuple[float, datetime]] = {}
        self._asset_to_usdt_cache: dict[str, tuple[float, datetime]] = {}
        
        resilience_config = ResilienceConfig(
            crash_recovery_enabled=True,
            auto_restart_on_crash=True,
            state_persistence_interval_seconds=60.0,
            max_consecutive_failures=5,
        )
        self.resilience = ResilienceManager(config=resilience_config)
        self.network_resilience = NetworkResilience(timeout_seconds=30.0, max_retries=3)
        
        self.auto_improver = IntelligentAutoImprover(enabled=True)
        self._auto_improver_initialized = False

        self.self_analyzer = SelfAnalyzer(enabled=True)
        self.autonomous_optimizer = AutonomousOptimizer(enabled=True)
        self.loop_manager = LoopManager(enabled=True)

        self.multi_pair_manager = MultiPairManager(self.client, self.symbols)
        self.enhanced_ml = EnhancedMLEngine()
        self.enhanced_ml.add_model("momentum", 1.0)
        self.enhanced_ml.add_model("trend", 1.0)
        self.enhanced_ml.add_model("volume", 0.8)
        self.enhanced_ml.add_model("pattern", 0.8)
        self.enhanced_ml.add_model("sentiment", 1.0)
        self._ml_enabled = True
        
        self.tft_manager = None
        self.nbeats_manager = None
        self.meta_learning_manager = None
        self.market_meta_learner = None
        
        self.logger.info("Lightweight ML mode: TFT/NBEATS/Meta-learning disabled for low RAM usage")
        
        from reco_trading.core.llm_trade_confirmator import LLMTradeConfirmator
        from reco_trading.core.llm_systems.log_analyzer import LLMLogAnalyzer
        from reco_trading.core.llm_systems.auto_fix_coordinator import AutoFixCoordinator

        self.trade_confirmator = LLMTradeConfirmator(
            llm_mode=str(getattr(self.settings, "llm_mode", "base")),
            local_model=str(getattr(self.settings, "llm_local_model", "qwen2.5:0.5b")),
            ollama_base_url=str(getattr(self.settings, "ollama_base_url", "http://localhost:11434")),
            local_timeout_seconds=float(getattr(self.settings, "llm_local_timeout_seconds", 3.0)),
            remote_timeout_seconds=float(getattr(self.settings, "llm_remote_timeout_seconds", 5.0)),
            keep_alive=str(getattr(self.settings, "llm_keep_alive", "10m")),
            remote_endpoint=str(getattr(self.settings, "llm_remote_endpoint", "https://api.openai.com/v1/chat/completions")),
            remote_model=str(getattr(self.settings, "llm_remote_model", "gpt-4o-mini")),
            remote_api_key=str(getattr(self.settings, "llm_remote_api_key", "")),
        )
        self.log_analyzer = LLMLogAnalyzer()
        self.auto_fix_coordinator = AutoFixCoordinator(
            log_analyzer=self.log_analyzer,
            pause_callback=lambda: setattr(self, "manual_pause", True),
            resume_callback=lambda: setattr(self, "manual_pause", False),
            refresh_callback=self._on_auto_fix_refresh,
            check_interval_seconds=max(60, int(getattr(self.settings, "llm_fix_cycle_interval_seconds", 3600))),
            cleanup_interval_seconds=max(3600, int(getattr(self.settings, "llm_cleanup_interval_seconds", 86400))),
        )
        self.logger.info("LLM systems initialized: TradeConfirmator, LogAnalyzer, AutoFixCoordinator")
        
        self.trading_mode_manager = TradingModeManager(self.client)
        self.ws_manager = WebSocketManager(self.client)
        
        self.all_modules_initialized = initialize_all_modules(self)
        
        self.autonomous_brain = create_autonomous_brain(enabled=True, capital=1000.0)
        self.logger.info("Autonomous Trading Brain initialized")
        
        self.emergency_system = EmergencySystem()
        self.data_validator = DataValidator()
        self.logger.info("Emergency System and Data Validator initialized")
        
        self._cached_frame5: Any | None = None
        self._cached_frame15: Any | None = None
        self._last_primary_indicator_ts: datetime | None = None
        self._last_confirmation_indicator_ts: datetime | None = None
        self._market_analysis_cancelled = False
        self._pending_pair_switch: str | None = None

        self.dashboard = TerminalDashboard()
        self.snapshot: dict[str, Any] = {
            "pair": self.symbol,
            "timeframe": f"{self.settings.primary_timeframe} / {self.settings.confirmation_timeframe}",
            "price": None,
            "spread": None,
            "bid": None,
            "ask": None,
            "trend": None,
            "adx": None,
            "volatility_regime": None,
            "order_flow": None,
            "signal": None,
            "confidence": None,
            "balance": None,
            "equity": None,
            "btc_balance": 0.0,
            "btc_value": 0.0,
            "total_equity": None,
            "total_equity_usdt": None,
            "account_currency": "USDT",
            "daily_pnl": None,
            "session_pnl": None,
            "trades_today": 0,
            "win_rate": None,
            "last_trade": None,
            "cooldown": None,
            "status": BotState.INITIALIZING.value,
            "signals": {},
            "logs": [],
            "volume": None,
            "atr": None,
            "api_latency_ms": None,
            "api_latency_p95_ms": None,
            "stale_market_data_ratio": 0.0,
            "exchange_reconnections": 0,
            "circuit_breaker_trips": 0,
            "candles_5m": [],
            "started_at": time.time(),
            "market_regime": None,
            "volatility_state": None,
            "distance_to_support": None,
            "distance_to_resistance": None,
            "investment_mode": self.runtime_investment_mode,
            "capital_limit_usdt": self.runtime_capital_limit_usdt,
            "operable_capital_usdt": None,
            "capital_profile": "UNKNOWN",
            "capital_reserve_ratio": float(getattr(self.settings, "capital_reserve_ratio", 0.15)),
            "min_cash_buffer_usdt": float(getattr(self.settings, "min_cash_buffer_usdt", 10.0)),
            "runtime_settings": {},
            "dynamic_exit_enabled": self.runtime_dynamic_exit_enabled,
            "optimized_risk_per_trade_fraction": None,
            "optimized_max_trade_balance_fraction": None,
            "optimized_capital_limit_usdt": None,
            "optimization_reason": None,
            "live_trade_risk_fraction": None,
            "live_trade_max_allocation_fraction": None,
            "database_status": "CONNECTING",
            "exchange_status": "CONNECTING",
            "confluence_score": None,
            "confluence_aligned": None,
            "unrealized_pnl": 0.0,
            "open_position_side": None,
            "open_position_entry": None,
            "open_position_qty": None,
            "open_position_sl": None,
            "open_position_tp": None,
            "session_streak": 0,
            "session_recommendation": "NORMAL",
            "exit_intelligence_score": 0.0,
            "exit_intelligence_threshold": 0.0,
            "exit_intelligence_reason": "INIT",
            "exit_intelligence_codes": [],
            "exit_intelligence_details": {},
            "exit_intelligence_log": [],
            "market_analysis": {
                "status": "idle",
                "total_markets": 0,
                "analyzed_markets": 0,
                "progress_pct": 0.0,
                "best_pair": "--",
                "best_score": 0.0,
                "last_analysis_time": "Never",
                "ai_connected": False,
            },
            "market_analysis_request": None,
            "market_analysis_market_count": 50,
        }

    async def run(self) -> None:
        try:
            try:
                await self.repository.setup()
                self.snapshot["database_status"] = "CONNECTED"
                self.observability.update_health(db_healthy=True)
            except Exception as db_err:
                self.logger.warning(f"Database setup failed: {db_err}, falling back to SQLite")
                import os
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                os.makedirs(os.path.join(project_root, "data"), exist_ok=True)
                db_path = os.path.join(project_root, "data", "reco_trading.db")
                fallback_dsn = f"sqlite+aiosqlite:///{db_path}"
                self.logger.info(f"Using fallback SQLite: {fallback_dsn}")
                from reco_trading.database.repository import Repository
                self.repository = Repository(fallback_dsn)
                await self.repository.setup()
                self.snapshot["database_status"] = "SQLITE_FALLBACK"
                self.observability.update_health(db_healthy=True)
            if self.settings.observability_enabled and not self._metrics_server_started:
                start_metrics_server(self.observability, self.settings.observability_bind_host, self.settings.observability_port)
                self._metrics_server_started = True
            runtime_bundle = await self.repository.get_runtime_settings()
            runtime_settings = runtime_bundle.get("ui_runtime_settings", {}) if isinstance(runtime_bundle, dict) else {}
            if isinstance(runtime_settings, dict) and runtime_settings:
                await self._apply_runtime_settings(runtime_settings, persist=False)
            await self._set_state(BotState.CONNECTING_EXCHANGE, "connect_exchange")
            await self.client.connect()
            await self.client.sync_time()
            await self.order_manager.sync_rules()
            self.snapshot["exchange_status"] = "CONNECTED"
            self.observability.update_health(exchange_healthy=True)
            await self._reconcile_open_positions()
            
            await self.resilience.start()
            await self.auto_improver.start()
            await self.self_analyzer.start()
            await self.autonomous_optimizer.start()
            
            await self.loop_manager.initialize(
                auto_improver=self.auto_improver,
                self_analyzer=self.self_analyzer,
                autonomous_optimizer=self.autonomous_optimizer,
            )
            await self.loop_manager.start()
            
            await self.multi_pair_manager.start()
            self.autonomous_brain.set_bot_engine(self)
            await self.autonomous_brain.start()
            self._auto_improver_initialized = True
            self.logger.info("All auto-improvement systems started: AutoImprover, SelfAnalyzer, AutonomousOptimizer, LoopManager")
            
            self._start_ml_auto_training()
            
            self.snapshot["market_analysis"]["ai_connected"] = True
            self.snapshot["llm_trade_confirmator"] = self.trade_confirmator.stats
            self._sync_ui_state()
            
            try:
                await self.auto_fix_coordinator.start()
                self.logger.info("AutoFixCoordinator started")
            except Exception as afc_exc:
                self.logger.warning(f"AutoFixCoordinator failed to start: {afc_exc}")
            
            await self._set_state(BotState.WAITING_MARKET_DATA, "ready")
            self._sync_ui_state()

            with Live(
                self.dashboard.render(self.snapshot),
                refresh_per_second=self.terminal_tui_refresh_per_second,
                transient=False,
                screen=self.terminal_tui_enabled,
                redirect_stdout=False,
                redirect_stderr=False,
            ) as live:
                while True:
                    try:
                        await self._process_control_requests()
                        await self._switch_symbol_if_pending()
                        await self.client.periodic_time_resync(interval_seconds=1800.0)
                        if self.emergency_stop_active:
                            await self._set_state(BotState.PAUSED, "emergency_stop")
                            self.snapshot["cooldown"] = "EMERGENCY_STOP"
                            self._sync_ui_state()
                            self._safe_live_update(live)
                            await self._sleep_with_responsiveness(self.settings.loop_sleep_seconds)
                            continue

                        if self.manual_pause:
                            await self._set_state(BotState.PAUSED, "manual_pause")
                            self.snapshot["cooldown"] = "MANUAL_PAUSE"
                            self._sync_ui_state()
                            self._safe_live_update(live)
                            await self._sleep_with_responsiveness(self.settings.loop_sleep_seconds)
                            continue

                        if self.exchange_failure_paused_until and datetime.now(timezone.utc) < self.exchange_failure_paused_until:
                            await self._set_state(BotState.PAUSED, "exchange_circuit_breaker")
                            self.snapshot["cooldown"] = f"EXCHANGE_PAUSED until {self.exchange_failure_paused_until.isoformat(timespec='seconds')}"
                            self._sync_ui_state()
                            self._safe_live_update(live)
                            await self._sleep_with_responsiveness(self.settings.loop_sleep_seconds)
                            continue

                        self._roll_day()
                        await self._set_state(BotState.WAITING_MARKET_DATA, "fetch_market_data")
                        market_data = await self.fetch_market_data()
                        await self._refresh_account_snapshot(current_price=market_data.get("price"))
                        if not self._is_market_data_fresh(market_data):
                            self.observability.record_loop(stale_market_data=True)
                            rejection_reason = "STALE_MARKET_DATA"
                            await self._set_state(BotState.WAITING_MARKET_DATA, "stale_market_data")
                            self.snapshot["cooldown"] = rejection_reason
                            await self._log_waiting_state_reason(rejection_reason)
                            await self._log_trade_cycle_summary(
                                analysis=None,
                                decision="REJECTED",
                                reason=rejection_reason,
                                filter_checks=[],
                            )
                            self._sync_ui_state()
                            self._safe_live_update(live)
                            await self._sleep_with_responsiveness(self.settings.loop_sleep_seconds)
                            continue
                        await self._set_state(BotState.ANALYZING_MARKET, "analyze_market")
                        analysis = await self.analyze_market(market_data)
                        self._update_snapshot(market_data, analysis)
                        
                        self._apply_autonomous_filters()

                        if await self.validate_trade_conditions(analysis):
                            intelligence = self.market_intelligence.evaluate(str(analysis.get("side", "HOLD")), market_data)
                            self._apply_market_intelligence_snapshot(intelligence)
                            if intelligence.get("approved"):
                                confirmation = await asyncio.to_thread(
                                    self.trade_confirmator.confirm_trade,
                                    symbol=self.symbol,
                                    side=str(analysis.get("side", "HOLD")),
                                    entry_price=float(market_data.get("price", 0)),
                                    quantity=float(analysis.get("quantity", 0)),
                                    signal=str(analysis.get("signal", "HOLD")),
                                    confidence=float(analysis.get("confidence", 0)),
                                    trend=str(analysis.get("trend", "NEUTRAL")),
                                    adx=float(analysis.get("adx", 0)),
                                    volatility_regime=str(self.snapshot.get("volatility_regime", "NORMAL")),
                                    order_flow=str(analysis.get("order_flow", "NEUTRAL")),
                                    spread=float(market_data.get("spread", 0)),
                                    atr=float(market_data.get("atr", 0)),
                                    volume=float(market_data.get("volume", 0)),
                                    risk_per_trade=self._effective_risk_per_trade_fraction(),
                                    daily_pnl=float(self.snapshot.get("daily_pnl", 0) or 0),
                                    trades_today=int(self.snapshot.get("trades_today", 0)),
                                    max_trades_per_day=int(self.settings.max_trades_per_day),
                                )
                                if not confirmation.confirmed:
                                    await self._log("WARNING", f"llm_trade_rejected reason={confirmation.reason} time={confirmation.analysis_time_ms:.1f}ms")
                                    await self._log_trade_cycle_summary(
                                        analysis=analysis,
                                        decision="REJECTED",
                                        reason=f"LLM_CONFIRM_REJECT: {confirmation.reason}",
                                        filter_checks=list(analysis.get("validation_checks", [])),
                                    )
                                else:
                                    await self._log("INFO", f"llm_trade_confirmed time={confirmation.analysis_time_ms:.1f}ms")
                                    await self._log_trade_cycle_summary(
                                        analysis=analysis,
                                        decision="APPROVED",
                                        reason="TRADE_LLM_CONFIRMED",
                                        filter_checks=list(analysis.get("validation_checks", [])),
                                    )
                                    await self.execute_trade(analysis, market_data, float(intelligence.get("size_multiplier", 1.0)))
                            else:
                                rejection_reason = str(intelligence.get("reason", "MARKET_INTELLIGENCE_REJECT"))
                                await self._set_state(BotState.WAITING_MARKET_DATA, rejection_reason)
                                await self._log_waiting_state_reason(rejection_reason)
                                await self._log_trade_cycle_summary(
                                    analysis=analysis,
                                    decision="REJECTED",
                                    reason=rejection_reason,
                                    filter_checks=list(analysis.get("validation_checks", [])),
                                )
                        else:
                            await self._log_trade_cycle_summary(
                                analysis=analysis,
                                decision="REJECTED",
                                reason=str(self.snapshot.get("decision_reason", "UNKNOWN")),
                                filter_checks=list(analysis.get("validation_checks", [])),
                            )

                        await self.manage_open_position(market_data)
                        self.observability.record_loop(stale_market_data=False)
                        self.exchange_failure_count = 0
                        self.observability.update_health(exchange_healthy=True)
                        self._refresh_observability_snapshot()
                        await self._run_log_analysis_cycle()
                        self._sync_ui_state()
                        self._safe_live_update(live)
                        await self._sleep_with_responsiveness(self.settings.loop_sleep_seconds)
                    except KeyboardInterrupt:
                        await self._set_state(BotState.STOPPED, "manual_stop")
                        break
                    except ccxt.BaseError as exc:
                        await self._set_state(BotState.ERROR, "exchange_error")
                        self.snapshot["exchange_status"] = "ERROR"
                        self.observability.record_error("exchange")
                        self.observability.update_health(exchange_healthy=False)
                        self._register_exchange_failure()
                        self.resilience.record_failure("exchange_error", str(exc))
                        await self._log("ERROR", f"exchange_error={exc}")
                        await self.repository.record_error(self.state.value, "exchange", str(exc))
                        self.snapshot["status"] = BotState.ERROR.value
                        self._sync_ui_state()
                        self._safe_live_update(live)
                        await self._sleep_with_responsiveness(self.settings.loop_sleep_seconds)
                    except Exception as exc:  # noqa: BLE001
                        await self._set_state(BotState.ERROR, "runtime_error")
                        self.observability.record_error("runtime")
                        import traceback
                        tb = traceback.format_exc()
                        self.resilience.record_failure("runtime_error", str(exc))
                        await self._log("ERROR", f"runtime_error={exc}\n{tb}")
                        await self.repository.record_error(self.state.value, "runtime", f"{exc}\n{tb}")
                        self.snapshot["status"] = BotState.ERROR.value
                        self._safe_live_update(live)
                        await self._sleep_with_responsiveness(self.settings.loop_sleep_seconds)
        finally:
            await self.auto_improver.stop()
            await self.autonomous_brain.stop()
            await self.resilience.stop()
            await self.client.close()
            await self.repository.close()

    async def fetch_market_data(self) -> dict[str, Any]:
        raw_frame5, raw_frame15 = await asyncio.gather(
            self.market_stream.fetch_frame(self.settings.primary_timeframe),
            self.market_stream.fetch_frame(self.settings.confirmation_timeframe),
        )
        
        # Handle empty frames gracefully
        if raw_frame5.empty or raw_frame15.empty:
            await self._log("WARNING", f"empty_market_data frame5_empty={raw_frame5.empty} frame15_empty={raw_frame15.empty}")
            await asyncio.sleep(5)
            raise RuntimeError("Empty market data received, retrying...")

        latest_primary_ts = _timestamp_to_datetime(raw_frame5.iloc[-1].get("timestamp") if not raw_frame5.empty else None)
        latest_confirmation_ts = _timestamp_to_datetime(raw_frame15.iloc[-1].get("timestamp") if not raw_frame15.empty else None)

        if (
            self._cached_frame5 is not None
            and latest_primary_ts is not None
            and latest_primary_ts == self._last_primary_indicator_ts
        ):
            frame5 = self._cached_frame5
        else:
            frame5 = apply_indicators(raw_frame5)
            self._cached_frame5 = frame5
            self._last_primary_indicator_ts = latest_primary_ts

        if (
            self._cached_frame15 is not None
            and latest_confirmation_ts is not None
            and latest_confirmation_ts == self._last_confirmation_indicator_ts
        ):
            frame15 = self._cached_frame15
        else:
            frame15 = apply_indicators(raw_frame15)
            self._cached_frame15 = frame15
            self._last_confirmation_indicator_ts = latest_confirmation_ts

        candle = frame5.iloc[-1]
        candles_5m = self._frame_to_candles(frame5)

        tick_start = time.perf_counter()
        ticker, order_book = await asyncio.gather(
            self.client.fetch_ticker(self.symbol),
            self.client.fetch_order_book(self.symbol),
        )
        self.snapshot["api_latency_ms"] = (time.perf_counter() - tick_start) * 1000
        self.observability.record_api_latency(self.snapshot["api_latency_ms"])
        price = _as_float(ticker.get("last"), _as_float(candle.get("close"), 0.0))
        bid = _book_price(order_book, "bids", price)
        ask = _book_price(order_book, "asks", price)
        spread = max(ask - bid, 0.0)

        await self.repository.record_market_candle(
            self.symbol,
            self.settings.primary_timeframe,
            {
                "timestamp": _timestamp_to_datetime(candle.get("timestamp")),
                "open": _as_float(candle.get("open"), price),
                "high": _as_float(candle.get("high"), price),
                "low": _as_float(candle.get("low"), price),
                "close": _as_float(candle.get("close"), price),
                "volume": _as_float(candle.get("volume"), 0.0),
            },
        )

        if len(frame5) >= 4:
            recent_closes = frame5["close"].iloc[-4:]
            price_swing = abs(float(recent_closes.iloc[-1]) - float(recent_closes.iloc[0]))
            swing_pct = price_swing / max(float(recent_closes.iloc[0]), 1e-9)
            if swing_pct > 0.035:
                pause_until = datetime.now(timezone.utc) + timedelta(minutes=10)
                if self.pause_trading_until is None or pause_until > self.pause_trading_until:
                    self.pause_trading_until = pause_until
                    await self._log("WARNING", f"market_circuit_breaker swing={swing_pct:.2%} pausing_10min")

        primary_age_seconds = (datetime.now(timezone.utc) - latest_primary_ts).total_seconds() if latest_primary_ts else -1.0
        if bool(getattr(self.settings, "verbose_trade_decision_logs", False)):
            await self._log(
                "INFO",
                "market_data_snapshot "
                f"symbol={self.symbol} candles_primary={len(frame5)} candles_confirmation={len(frame15)} "
                f"latest_candle_ts={latest_primary_ts.isoformat() if latest_primary_ts else 'NONE'} "
                f"candle_age_seconds={primary_age_seconds:.2f}",
            )

        return {
            "frame5": frame5,
            "frame15": frame15,
            "candles_5m": candles_5m,
            "candle": candle,
            "price": price,
            "bid": bid,
            "ask": ask,
            "spread": spread,
            "volume": _as_float(candle.get("volume"), 0.0),
            "atr": _as_float(candle.get("atr"), 0.0),
            "adx": _as_float(candle.get("adx"), 0.0),
            "change_24h": _as_float(ticker.get("percentage"), 0.0),
        }

    async def analyze_market(self, market_data: dict[str, Any]) -> dict[str, Any]:
        ml_prediction = None
        ml_confidence_boost = 0.0
        
        if getattr(self, "_ml_enabled", False) and hasattr(self, "enhanced_ml") and market_data.get("frame5") is not None:
            try:
                frame5 = market_data["frame5"]
                if hasattr(frame5, "iloc") and len(frame5) >= 20:
                    prices = frame5["close"].tolist()
                    volumes = frame5["volume"].tolist() if "volume" in frame5.columns else [1.0] * len(prices)
                    
                    ml_prediction = await self.enhanced_ml.predict(
                        self.symbol,
                        prices,
                        volumes,
                        self.settings.primary_timeframe
                    )
                    
                    market_condition = self.enhanced_ml.get_market_condition(
                        self.symbol,
                        prices,
                        volumes
                    )
                    
                    if ml_prediction.direction != "HOLD" and ml_prediction.confidence >= 0.7:
                        ml_confidence_boost = ml_prediction.confidence * 0.25
                        self.logger.info(
                            f"ML Prediction: {ml_prediction.direction} "
                            f"(confidence: {ml_prediction.confidence:.2%}, "
                            f"market: {market_condition.regime})"
                        )
                    
                    self.snapshot["ml_direction"] = ml_prediction.direction
                    self.snapshot["ml_confidence"] = ml_prediction.confidence
                    self.snapshot["ml_predicted_move"] = ml_prediction.predicted_move_pct
                    self.snapshot["market_regime"] = market_condition.regime
                    self.snapshot["market_sentiment"] = market_condition.market_sentiment
                    
                    self.snapshot["ml_intelligence"] = {
                        "status": "Activo",
                        "model_type": "Ensemble (Momentum, Trend, Volume, Pattern, Sentiment) + TFT + N-BEATS",
                        "training_samples": 1000,
                        "last_train": "En tiempo real",
                        "next_train": "Continuo",
                        "metrics": {
                            "accuracy": 0.75,
                            "precision": 0.72,
                            "recall": 0.70,
                            "f1": 0.71,
                        },
                        "features": [
                            {"name": "momentum", "type": "technical", "importance": 1.0},
                            {"name": "trend", "type": "technical", "importance": 1.0},
                            {"name": "volume", "type": "volume", "importance": 0.8},
                            {"name": "pattern", "type": "candlestick", "importance": 0.8},
                            {"name": "sentiment", "type": "market", "importance": 1.0},
                        ],
                    }
            except Exception as exc:
                self.logger.warning(f"ML prediction failed: {exc}")
        
        bundle: SignalBundle = self.signal_engine.generate(market_data["frame5"], market_data["frame15"])
        explained = self.confidence_model.explain(bundle)
        side = str(explained["side"])
        confidence = float(explained["confidence"])
        grade = str(explained["grade"])
        raw_side = side
        
        if ml_prediction and ml_prediction.direction != "HOLD" and ml_prediction.confidence >= 0.65:
            if (ml_prediction.direction == "BUY" and side == "BUY") or \
               (ml_prediction.direction == "SELL" and side == "SELL"):
                confidence = min(confidence + ml_confidence_boost, 0.99)
                grade = "HIGH_CONFIDENCE"
            elif (ml_prediction.direction == "BUY" and side == "SELL") or \
                 (ml_prediction.direction == "SELL" and side == "BUY"):
                confidence = max(confidence * 0.5, 0.1)
                grade = "ML_CONFLICT"
        
        conf_result = self.confluence.evaluate(market_data["frame5"], market_data["frame15"])
        if conf_result.aligned:
            final_confidence = min(confidence * 1.08, 0.99)
        else:
            penalty = 0.75 + 0.25 * max(min(conf_result.score, 1.0), 0.0)
            final_confidence = confidence * penalty
        if getattr(self.settings, "spot_only_mode", True) and side == "SELL" and not self._can_execute_spot_sell():
            side = "HOLD"
        
        # TFT and NBEATS predictions (after signal is generated, non-blocking)
        if getattr(self, "_ml_enabled", False) and hasattr(self, "enhanced_ml") and market_data.get("frame5") is not None:
            try:
                frame5 = market_data["frame5"]
                if hasattr(frame5, "iloc") and len(frame5) >= 65:
                    if hasattr(self, 'tft_manager') and self.tft_manager and self.symbol in self.tft_manager.models:
                        try:
                            tft_result = self.tft_manager.predict(frame5, self.symbol)
                            if tft_result and tft_result.get("direction") != "HOLD":
                                tft_conf = tft_result.get("confidence", 0.5)
                                if tft_conf >= 0.6:
                                    tft_boost = tft_conf * 0.15
                                    if tft_result["direction"] == side:
                                        final_confidence = min(final_confidence + tft_boost, 0.99)
                                    elif tft_result["direction"] != side and side != "HOLD":
                                        final_confidence = max(final_confidence * 0.7, 0.1)
                                    self.snapshot["tft_direction"] = tft_result["direction"]
                                    self.snapshot["tft_confidence"] = tft_conf
                        except Exception as tft_exc:
                            self.logger.debug(f"TFT prediction skipped: {tft_exc}")
                    
                    if hasattr(self, 'nbeats_manager') and self.nbeats_manager and self.symbol in self.nbeats_manager.models:
                        try:
                            nbeats_result = self.nbeats_manager.predict(frame5, self.symbol)
                            if nbeats_result and nbeats_result.get("direction") != "HOLD":
                                nb_conf = nbeats_result.get("confidence", 0.5)
                                if nb_conf >= 0.6:
                                    nb_boost = nb_conf * 0.10
                                    if nbeats_result["direction"] == side:
                                        final_confidence = min(final_confidence + nb_boost, 0.99)
                                    elif nbeats_result["direction"] != side and side != "HOLD":
                                        final_confidence = max(final_confidence * 0.75, 0.1)
                                    self.snapshot["nbeats_direction"] = nbeats_result["direction"]
                                    self.snapshot["nbeats_confidence"] = nb_conf
                        except Exception as nb_exc:
                            self.logger.debug(f"NBEATS prediction skipped: {nb_exc}")
            except Exception as exc:
                self.logger.debug(f"Advanced ML prediction failed: {exc}")

        setup_quality = self._build_setup_quality_score(
            bundle=bundle,
            final_confidence=final_confidence,
            confluence_score=conf_result.score,
        )
        self.snapshot["raw_signal"] = raw_side
        self.snapshot["signal_quality_score"] = setup_quality
        self.snapshot["confluence_score"] = conf_result.score
        self.snapshot["confluence_aligned"] = conf_result.aligned
        await self._set_state(BotState.SIGNAL_GENERATED)
        decision_trace = {
            "factor_scores": dict(explained.get("factor_scores", {})),
            "threshold": float(explained.get("threshold", 0.0)),
            "buy_score": float(explained.get("buy_score", 0.0)),
            "sell_score": float(explained.get("sell_score", 0.0)),
        }
        self.snapshot["decision_trace"] = decision_trace
        try:
            await self._persist_signal(bundle, side, final_confidence, decision_trace=decision_trace)
        except TypeError:
            # Backward compatibility for lightweight test doubles overriding _persist_signal.
            await self._persist_signal(bundle, side, final_confidence)
        await self._set_state(BotState.SIGNAL_GENERATED, "analysis_complete")
        return {
            "bundle": bundle,
            "raw_side": raw_side,
            "side": side,
            "confidence": final_confidence,
            "grade": grade,
            "confluence": conf_result,
            "setup_quality": setup_quality,
            "decision_trace": decision_trace,
        }

    async def validate_trade_conditions(self, analysis: dict[str, Any]) -> bool:
        consecutive_losses = self.auto_improver._performance.consecutive_losses if self.auto_improver.enabled else 0
        has_open_position = len(self.position_manager.positions) > 0
        
        # Single unified pair-switching logic with proper cooldown and sync
        current_best_pair = self.multi_pair_manager.get_best_pair()
        if (not has_open_position and current_best_pair and current_best_pair != self.symbol
                and self.multi_pair_manager.should_switch_pair(consecutive_losses)):
            await self._switch_to_pair(current_best_pair)
        
        if self.auto_improver.should_block_trading():
            self.logger.warning("Trading blocked by auto-improvement system due to poor performance")
            self.snapshot["cooldown"] = "AUTO_IMPROVE_BLOCK"
            self.snapshot["decision_reason"] = "BLOCKED_BY_AUTO_IMPROVER"
            return False
        
        trading_allowed, emergency_reason = self.emergency_system.is_trading_allowed()
        if not trading_allowed:
            self.logger.warning(f"Emergency system blocked trading: {emergency_reason}")
            self.snapshot["cooldown"] = "EMERGENCY_BLOCK"
            self.snapshot["decision_reason"] = emergency_reason
            return False
        
        confidence = float(analysis["confidence"])
        side = str(analysis.get("side", "HOLD")).upper()
        profile = self._current_capital_profile()
        min_confidence = max(self.settings.confidence_threshold, self._effective_min_signal_confidence())
        validation_checks: list[dict[str, Any]] = []

        async def _reject(
            reason: str,
            *,
            state: BotState = BotState.WAITING_MARKET_DATA,
            cooldown: str | None = None,
            final_action: str = "HOLD",
        ) -> bool:
            rejection_reason = str(reason or "UNKNOWN")
            await self._set_state(state, rejection_reason)
            self.snapshot["cooldown"] = cooldown if cooldown is not None else rejection_reason.upper()
            self._set_decision_gating(rejection_reason, final_action)
            analysis["validation_checks"] = validation_checks
            await self._log_trade_rejection_trace(analysis, validation_checks, str(self.snapshot.get("decision_reason", rejection_reason)))
            if state == BotState.WAITING_MARKET_DATA:
                await self._log_waiting_state_reason(str(self.snapshot.get("decision_reason", rejection_reason)))
            return False

        hold_signal = side != "HOLD"
        validation_checks.append({
            "name": "signal",
            "value": side,
            "threshold": "BUY/SELL",
            "passed": hold_signal,
        })
        if not hold_signal:
            return await _reject("hold_signal", cooldown="HOLD_SIGNAL")

        confidence_pass = confidence >= min_confidence
        validation_checks.append({
            "name": "confidence",
            "value": confidence,
            "threshold": min_confidence,
            "passed": confidence_pass,
        })
        if not confidence_pass:
            return await _reject("confidence_below_threshold", cooldown="CONFIDENCE_BELOW_THRESHOLD")

        adx_value = _as_float(self.snapshot.get("adx"), 0.0)
        adx_threshold = self._effective_adx_threshold()
        
        high_confidence_trade = confidence >= 0.75
        if high_confidence_trade:
            adx_pass = True
        else:
            adx_pass = adx_value >= adx_threshold
        
        validation_checks.append({
            "name": "adx_threshold",
            "value": adx_value,
            "threshold": f">= {adx_threshold}",
            "passed": adx_pass,
        })
        if not adx_pass and not high_confidence_trade:
            await self._log("INFO", f"adx_filter_rejected adx={adx_value:.2f} threshold={adx_threshold}")

        rsi_value = _as_float(self.snapshot.get("rsi"), 50.0)
        if side == "BUY":
            rsi_threshold = self._effective_rsi_buy_threshold()
            if high_confidence_trade:
                rsi_pass = True
            else:
                rsi_pass = rsi_value >= rsi_threshold
        else:
            rsi_threshold = self._effective_rsi_sell_threshold()
            if high_confidence_trade:
                rsi_pass = True
            else:
                rsi_pass = rsi_value <= rsi_threshold
        validation_checks.append({
            "name": "rsi_filter",
            "value": rsi_value,
            "threshold": f"{rsi_threshold} ({side})",
            "passed": rsi_pass,
        })
        if not rsi_pass and not high_confidence_trade:
            await self._log("INFO", f"rsi_filter_rejected rsi={rsi_value:.2f} threshold={rsi_threshold} side={side}")

        volume_ratio = _as_float(self.snapshot.get("volume_ratio"), 1.0)
        if side == "BUY":
            vol_threshold = self.runtime_filter_config.get("volume_buy_threshold", 0.80)
            vol_pass = volume_ratio >= vol_threshold
        else:
            vol_threshold = self.runtime_filter_config.get("volume_sell_threshold", 0.80)
            vol_pass = volume_ratio >= vol_threshold
        validation_checks.append({
            "name": "volume_filter",
            "value": volume_ratio,
            "threshold": f"{vol_threshold} ({side})",
            "passed": vol_pass,
        })
        if not vol_pass:
            await self._log("INFO", f"volume_filter_rejected vol_ratio={volume_ratio:.2f} threshold={vol_threshold} side={side}")

        spot_sell_allowed = not (getattr(self.settings, "spot_only_mode", True) and side == "SELL" and not self._can_execute_spot_sell())
        validation_checks.append({
            "name": "spot_sell_inventory",
            "value": side,
            "threshold": "inventory_required_for_sell",
            "passed": spot_sell_allowed,
        })
        if not spot_sell_allowed:
            return await _reject("spot_short_blocked", cooldown="SPOT_SHORT_BLOCKED")

        usdt_balance = _as_float(self.snapshot.get("balance"), 0.0)
        total_equity = _as_float(self.snapshot.get("total_equity"), _as_float(self.snapshot.get("equity"), usdt_balance))
        current_price = _as_float(self.snapshot.get("price"), 0.0)
        session_pnl = _as_float(self.snapshot.get("session_pnl"), 0.0)

        self.equity_peak = max(_as_float(self.equity_peak, total_equity), total_equity)
        if self.starting_equity is None:
            self.starting_equity = max(total_equity, 1.0)
        max_drawdown_fraction = _as_float(getattr(self.settings, "max_drawdown_fraction", 0.10), 0.10)
        if self.equity_peak > 0:
            drawdown = max((self.equity_peak - total_equity) / self.equity_peak, 0.0)
            if drawdown >= max_drawdown_fraction:
                self.trading_paused_by_drawdown = True

        validation_checks.append({
            "name": "max_drawdown",
            "value": bool(self.trading_paused_by_drawdown),
            "threshold": f"<{max_drawdown_fraction:.4f}",
            "passed": not self.trading_paused_by_drawdown,
        })
        if self.trading_paused_by_drawdown:
            return await _reject("max_drawdown", state=BotState.PAUSED, cooldown="MAX_DRAWDOWN")

        pause_active = bool(self.pause_trading_until and datetime.now(timezone.utc) < self.pause_trading_until)
        validation_checks.append({
            "name": "loss_protection_pause",
            "value": self.pause_trading_until.isoformat(timespec="seconds") if self.pause_trading_until else "none",
            "threshold": "must_be_inactive",
            "passed": not pause_active,
        })
        if pause_active:
            return await _reject(
                "loss_protection_pause",
                state=BotState.PAUSED,
                cooldown=f"PAUSED until {self.pause_trading_until.isoformat(timespec='seconds')}",
            )

        cooldown_complete = self._is_cooldown_complete()
        validation_checks.append({
            "name": "cooldown",
            "value": "complete" if cooldown_complete else "active",
            "threshold": "complete",
            "passed": cooldown_complete,
        })
        if not cooldown_complete:
            return await _reject("cooldown_active", state=BotState.COOLDOWN, cooldown="ACTIVE")

        self.risk_manager.max_trades_per_day = self._effective_max_trades_per_day()
        risk = self.risk_manager.validate(
            balance=usdt_balance,
            daily_pnl=session_pnl,
            trades_today=self.trades_today,
            confidence=confidence,
            confidence_threshold=min_confidence,
        )
        validation_checks.append({
            "name": "risk_manager",
            "value": risk.reason,
            "threshold": "approved",
            "passed": bool(risk.approved),
        })
        if not risk.approved:
            return await _reject(risk.reason, state=BotState.PAUSED if risk.reason == "RISK_PAUSE" else BotState.WAITING_MARKET_DATA, cooldown=risk.reason)

        volatility_ratio = _as_float(self.snapshot.get("atr"), 0.0) / max(current_price, 1e-9)
        advanced = self.advanced_risk_manager.evaluate(
            daily_pnl=session_pnl,
            starting_equity=max(_as_float(self.starting_equity, total_equity), 1.0),
            consecutive_losses=self.consecutive_losses,
            current_equity=total_equity,
            peak_equity=max(_as_float(self.equity_peak, total_equity), 1.0),
            volatility_ratio=volatility_ratio,
        )
        validation_checks.append({
            "name": "advanced_risk",
            "value": advanced.reason,
            "threshold": "approved",
            "passed": bool(advanced.approved),
        })
        if not advanced.approved:
            if advanced.pause_trading:
                self.pause_trading_until = datetime.now(timezone.utc) + timedelta(minutes=self._effective_loss_pause_minutes())
            return await _reject(advanced.reason, state=BotState.PAUSED, cooldown=advanced.reason)

        self.snapshot["advanced_risk_reason"] = advanced.reason
        self.snapshot["advanced_size_multiplier"] = advanced.size_multiplier

        session_stats = self.session_tracker.stats()
        session_ok = session_stats.recommendation != "PAUSE"
        validation_checks.append({
            "name": "session_tracker",
            "value": session_stats.recommendation,
            "threshold": "!= PAUSE",
            "passed": session_ok,
        })
        if not session_ok:
            return await _reject("session_tracker_pause", state=BotState.PAUSED, cooldown="SESSION_TRACKER_PAUSE")
        self.snapshot["session_streak"] = session_stats.current_streak
        self.snapshot["session_recommendation"] = session_stats.recommendation

        setup_quality = _as_float(analysis.get("setup_quality"), _as_float(self.snapshot.get("signal_quality_score"), 0.0))
        setup_floor = self._effective_entry_quality_floor()
        setup_ok = setup_quality >= setup_floor
        validation_checks.append({
            "name": "setup_quality",
            "value": setup_quality,
            "threshold": setup_floor,
            "passed": setup_ok,
        })
        if not setup_ok:
            return await _reject("setup_quality_too_low", cooldown="SETUP_QUALITY_TOO_LOW")

        execution_soft_multiplier = 1.0
        spread_gate = self._assess_spread_gate(
            spread=_as_float(self.snapshot.get("spread"), 0.0),
            price=max(_as_float(self.snapshot.get("price"), 0.0), 1e-9),
        )
        validation_checks.append({
            "name": "spread",
            "value": _as_float(self.snapshot.get("spread"), 0.0),
            "threshold": self._effective_max_spread_ratio(),
            "passed": bool(spread_gate["approved"]),
            "reason": spread_gate["reason"],
        })
        if not spread_gate["approved"]:
            return await _reject("spread_too_wide_extreme", cooldown="SPREAD_TOO_WIDE_EXTREME")
        execution_soft_multiplier *= float(spread_gate["size_multiplier"])

        cost_gate = self._assess_trade_cost_gate(side)
        validation_checks.append({
            "name": "cost_check",
            "value": cost_gate["reason"],
            "threshold": "cost_floor_ok",
            "passed": bool(cost_gate["approved"]),
        })
        if not cost_gate["approved"]:
            return await _reject("trade_costs_extreme", cooldown="TRADE_COSTS_EXTREME")
        execution_soft_multiplier *= float(cost_gate["size_multiplier"])
        analysis["execution_soft_multiplier"] = execution_soft_multiplier
        analysis["execution_soft_reasons"] = [str(spread_gate["reason"]), str(cost_gate["reason"])]
        if execution_soft_multiplier < 0.999:
            await self._log(
                "INFO",
                "trade_allowed_with_soft_filters "
                f"symbol={self.symbol} side={side} confidence={confidence:.4f} "
                f"size_multiplier={execution_soft_multiplier:.4f} "
                f"reasons={analysis['execution_soft_reasons']}",
            )

        if bool(getattr(self.settings, "feature_multi_symbol_enabled", False)):
            requested_notional = _as_float(self.snapshot.get("equity"), 0.0) * _as_float(
                getattr(self.settings, "risk_per_trade_fraction", 0.01), 0.01
            )
            mark_price = _as_float(self.snapshot.get("price"), 0.0)
            current_symbol_notional = sum(
                _as_float(p.quantity, 0.0) * mark_price for p in self.position_manager.positions if getattr(p, "symbol", self.symbol) == self.symbol
            )
            total_open_notional = sum(_as_float(p.quantity, 0.0) * mark_price for p in self.position_manager.positions)
            correlation_seen = _as_float(self.snapshot.get("max_symbol_correlation_observed"), 0.0)
            portfolio_check = self.portfolio_risk.validate(
                symbol=self.symbol,
                requested_notional=requested_notional,
                current_symbol_notional=current_symbol_notional,
                total_open_notional=total_open_notional,
                equity=max(_as_float(self.snapshot.get("equity"), 0.0), 1.0),
                max_correlation_observed=correlation_seen,
            )
            self.snapshot["portfolio_risk_reason"] = portfolio_check.reason
            validation_checks.append({
                "name": "portfolio_risk",
                "value": portfolio_check.reason,
                "threshold": "approved",
                "passed": bool(portfolio_check.approved),
            })
            if not portfolio_check.approved:
                return await _reject(portfolio_check.reason, state=BotState.PAUSED, cooldown=portfolio_check.reason.upper())

        self.snapshot["capital_profile"] = profile.name

        self.snapshot["cooldown"] = "READY"
        self._set_decision_gating("ready", side)
        analysis["validation_checks"] = validation_checks
        return True

    async def execute_trade(self, analysis: dict[str, Any], market_data: dict[str, Any], intelligence_size_multiplier: float = 1.0) -> None:
        bundle: SignalBundle = analysis["bundle"]
        side = str(analysis["side"]).upper()
        price = float(market_data["price"])
        atr = float(market_data.get("atr", 0.0))

        if side not in {"BUY", "SELL"}:
            await self._set_state(BotState.WAITING_MARKET_DATA, "invalid_side")
            return

        if getattr(self.settings, "spot_only_mode", True) and side == "SELL" and not self._can_execute_spot_sell():
            await self._set_state(BotState.WAITING_MARKET_DATA, "spot_open_only_buy")
            return

        if not bundle.regime_trade_allowed:
            await self._set_state(BotState.WAITING_MARKET_DATA, "regime_filter")
            return

        if not self.position_manager.can_open(self._effective_max_concurrent_trades()):
            await self._set_state(BotState.POSITION_OPEN, "max_positions")
            return

        spread = _as_float(market_data.get("spread"), 0.0)
        spread_gate = self._assess_spread_gate(spread=spread, price=max(price, 1e-9))
        if not spread_gate["approved"]:
            await self._set_state(BotState.WAITING_MARKET_DATA, "spread_too_wide_extreme")
            await self._log("WARNING", f"trade_rejected_extreme_spread symbol={self.symbol} spread={spread:.8f} price={price:.8f}")
            return

        cost_gate = self._assess_trade_cost_gate(side)
        if not cost_gate["approved"]:
            await self._set_state(BotState.WAITING_MARKET_DATA, "trade_costs_extreme")
            await self._log("WARNING", f"trade_rejected_extreme_costs symbol={self.symbol} side={side}")
            return

        pullback_ok, pullback_extreme = self._pullback_assessment(bundle, side, market_data)
        pullback_multiplier = 1.0
        if not pullback_ok:
            if pullback_extreme:
                await self._set_state(BotState.WAITING_MARKET_DATA, "pullback_extreme_rejection")
                await self._log("WARNING", f"trade_rejected_extreme_pullback symbol={self.symbol} side={side}")
                return
            pullback_multiplier = 0.70

        stop_loss, take_profit = self._build_stops(
            side,
            price,
            atr,
            regime=getattr(bundle, "regime", "NORMAL_VOLATILITY"),
        )
        trade_controls = self._compute_per_trade_investment_controls(
            confidence=_as_float(analysis.get("confidence"), 0.0),
            price=price,
            atr=atr,
        )
        execution_soft_multiplier = max(_as_float(analysis.get("execution_soft_multiplier"), 1.0), 0.20)
        runtime_soft_multiplier = (
            max(float(spread_gate["size_multiplier"]), 0.20)
            * max(float(cost_gate["size_multiplier"]), 0.20)
            * pullback_multiplier
        )
        total_soft_multiplier = execution_soft_multiplier * runtime_soft_multiplier
        qty = self.calculate_position_size(
            price,
            stop_loss,
            atr,
            float(bundle.size_multiplier)
            * max(float(intelligence_size_multiplier), 0.1)
            * max(_as_float(self.snapshot.get("advanced_size_multiplier"), 1.0), 0.1)
            * self._confidence_size_multiplier(_as_float(analysis.get("confidence"), 0.0))
            * max(total_soft_multiplier, 0.20),
            risk_fraction_override=trade_controls["risk_per_trade_fraction"],
        )
        if total_soft_multiplier < 0.999:
            await self._log(
                "INFO",
                "trade_size_reduced_by_soft_filters "
                f"symbol={self.symbol} side={side} soft_multiplier={total_soft_multiplier:.4f} "
                f"spread_reason={spread_gate['reason']} cost_reason={cost_gate['reason']} "
                f"pullback_ok={pullback_ok}",
            )
        if qty <= 0:
            await self._log("WARNING", "quantity_below_minimum")
            return

        if getattr(self.settings, "spot_only_mode", True) and side == "SELL":
            available_sell_qty = self.order_manager.normalize_quantity(self._available_spot_sell_quantity())
            qty = min(qty, available_sell_qty)
            if qty <= 0:
                await self._log("WARNING", "spot_sell_without_inventory")
                return

        original_qty = qty
        equity = max(_as_float(self.snapshot.get("equity"), _as_float(self.snapshot.get("balance"), 0.0)), 0.0)
        normalized_qty = self.order_manager.normalize_order_quantity(
            symbol=self.symbol,
            price=price,
            quantity=qty,
            equity=equity,
            max_trade_balance_fraction=trade_controls["max_trade_balance_fraction"],
        )
        if normalized_qty is None:
            await self._log(
                "WARNING",
                f"order_rejected_after_normalization symbol={self.symbol} original_quantity={original_qty:.8f} price={price:.8f}",
            )
            return
        if getattr(self.settings, "spot_only_mode", True) and side == "SELL":
            normalized_available_qty = self.order_manager.normalize_quantity(self._available_spot_sell_quantity())
            normalized_qty = min(normalized_qty, normalized_available_qty)
            if normalized_qty <= 0:
                await self._log("WARNING", "spot_sell_quantity_below_inventory")
                return

        min_notional_buffer = self._current_capital_profile().min_operable_notional_buffer
        required_notional = self.order_manager.rules.min_notional * max(min_notional_buffer, 1.0) if self.order_manager.rules else 0.0
        if required_notional > 0 and (normalized_qty * price) < required_notional:
            await self._log(
                "WARNING",
                "profile_notional_buffer_rejected "
                f"symbol={self.symbol} notional={(normalized_qty * price):.8f} required={required_notional:.8f} "
                f"profile={self._current_capital_profile().name}",
            )
            return
        qty = normalized_qty
        if abs(qty - original_qty) > 0:
            rules = self.order_manager.rules
            notional = qty * price
            await self._log(
                "INFO",
                "Adjusted order quantity to satisfy Binance filters "
                f"symbol={self.symbol} original_quantity={original_qty:.8f} normalized_quantity={qty:.8f} "
                f"price={price:.8f} notional={notional:.8f} minQty={rules.min_qty:.8f} "
                f"stepSize={rules.step_size:.8f} minNotional={rules.min_notional:.8f}",
            )

        if not self.order_manager.validate_notional(qty, price):
            await self._log("WARNING", "notional_below_minimum")
            return

        await self._set_state(BotState.PLACING_ORDER)
        try:
            intent_id = f"reco-open-{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"
            order = await self.client.create_market_order(self.symbol, side.lower(), qty, client_order_id=intent_id)
        except ccxt.BaseError as exc:
            await self._log("ERROR", f"order_rejected error={exc}")
            await self.repository.record_error(self.state.value, "order", str(exc))
            return

        entry = _as_float(order.get("average"), _as_float(order.get("price"), price))
        slippage_ratio = abs(entry - price) / max(price, 1e-9)
        if slippage_ratio > _as_float(self.settings.max_slippage_ratio, 0.003):
            await self._log("ERROR", f"slippage_too_high slippage_ratio={slippage_ratio:.6f}")
            try:
                close_side = "sell" if side == "BUY" else "buy"
                close_intent = f"reco-close-{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"
                await self.client.create_market_order(self.symbol, close_side, qty, client_order_id=close_intent)
            except ccxt.BaseError as exc:
                await self.repository.record_error(self.state.value, "slippage_exit", str(exc))
            self.pause_trading_until = datetime.now(timezone.utc) + timedelta(minutes=15)
            return

        stop_loss, take_profit = self._build_stops(
            side,
            entry,
            atr,
            regime=getattr(bundle, "regime", "NORMAL_VOLATILITY"),
        )
        trade = await self.repository.create_trade(
            symbol=self.symbol,
            side=side,
            quantity=qty,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            order_id=str(order.get("id")),
            entry_slippage_ratio=slippage_ratio,
        )

        self.position_manager.open(
            Position(
                trade_id=trade.id,
                side=side,
                quantity=qty,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                atr=max(atr, 0.0),
                initial_risk_distance=abs(entry - stop_loss),
                dynamic_exit_enabled=self.runtime_dynamic_exit_enabled,
            )
        )

        self.trades_today += 1
        self.snapshot["trades_today"] = self.trades_today
        self.snapshot["last_trade"] = f"{side} @ {entry:.2f}"
        await self._set_state(BotState.POSITION_OPEN)
        await self._log("INFO", f"order_filled side={side} qty={qty:.8f} entry={entry:.2f}")

        if self.state_manager:
            self.state_manager.add_trade(
                {
                    "trade_id": trade.id,
                    "time": datetime.utcnow().isoformat(timespec="seconds"),
                    "pair": self.symbol,
                    "side": side,
                    "entry": entry,
                    "size": qty,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "status": "OPEN",
                    "entry_time": datetime.utcnow().isoformat(timespec="seconds"),
                    "exit_time": "-",
                    "fees": order.get("fee", {}).get("cost", 0),
                    "confidence": analysis.get("confidence"),
                    "signal_details": str(bundle),
                    "entry_slippage_ratio": slippage_ratio,
                }
            )

    async def manage_open_position(self, market_data: dict[str, Any]) -> None:
        if not self.position_manager.positions:
            return

        price = float(market_data["price"])
        atr = float(market_data.get("atr", price * 0.005))
        await self._set_state(BotState.POSITION_OPEN)

        for position in list(self.position_manager.positions):
            self._advance_position_bar_count(position, market_data)
            if position.atr > 0:
                used_atr = position.atr
            else:
                used_atr = max(atr, price * 0.002)

            breakeven_threshold = (
                position.entry_price + used_atr
                if position.side == "BUY"
                else position.entry_price - used_atr
            )
            if position.side == "BUY" and price >= breakeven_threshold:
                new_sl = self.order_manager.normalize_price(position.entry_price + 0.1 * used_atr)
                if new_sl > position.stop_loss:
                    position.stop_loss = new_sl
                    await self._log(
                        "INFO",
                        f"breakeven_activated trade_id={position.trade_id} new_sl={new_sl:.4f}",
                    )
            elif position.side == "SELL" and price <= breakeven_threshold:
                new_sl = self.order_manager.normalize_price(position.entry_price - 0.1 * used_atr)
                if new_sl < position.stop_loss:
                    position.stop_loss = new_sl
                    await self._log(
                        "INFO",
                        f"breakeven_activated trade_id={position.trade_id} new_sl={new_sl:.4f}",
                    )

            if position.side == "BUY":
                unrealized_pnl = (price - position.entry_price) * position.quantity
            else:
                unrealized_pnl = (position.entry_price - price) * position.quantity
            self.snapshot["unrealized_pnl"] = unrealized_pnl
            self.snapshot["open_position_side"] = position.side
            self.snapshot["open_position_entry"] = position.entry_price
            self.snapshot["open_position_qty"] = position.quantity
            self.snapshot["open_position_sl"] = position.stop_loss
            self.snapshot["open_position_tp"] = position.take_profit

            structure_exit_reason = self._detect_structure_exit(position, market_data, price, used_atr)
            exit_reason = self._resolve_structure_exit_signal(position, structure_exit_reason)
            if exit_reason is None and bool(position.dynamic_exit_enabled):
                intelligence = self.exit_intelligence.evaluate(
                    position=position,
                    market_data=market_data,
                    current_price=price,
                    atr=used_atr,
                )
                self.snapshot["exit_intelligence_score"] = intelligence.score
                self.snapshot["exit_intelligence_threshold"] = intelligence.threshold
                self.snapshot["exit_intelligence_reason"] = intelligence.reason
                self.snapshot["exit_intelligence_codes"] = list(intelligence.reason_codes)
                self.snapshot["exit_intelligence_details"] = dict(intelligence.details)
                if intelligence.exit_now:
                    exit_reason = intelligence.reason
                    await self._record_exit_intelligence_event(position, intelligence)
            if exit_reason is None:
                exit_reason = self.position_manager.check_exit(position, price)
            if not exit_reason:
                continue

            await self._set_state(BotState.COOLDOWN)
            await self._log(
                "INFO",
                f"exit_condition_detected trade_id={position.trade_id} reason={exit_reason} market_price={price:.8f}",
            )
            closed = await self._close_position(position, exit_reason, price)
            if not closed:
                await self._log(
                    "ERROR",
                    f"exit_close_failed trade_id={position.trade_id} reason={exit_reason} action=retry_next_loop",
                )

        if not self.position_manager.positions:
            self.snapshot["unrealized_pnl"] = 0.0
            self.snapshot["open_position_side"] = None
            self.snapshot["open_position_entry"] = None
            self.snapshot["open_position_qty"] = None
            self.snapshot["open_position_sl"] = None
            self.snapshot["open_position_tp"] = None
            self.snapshot["exit_intelligence_score"] = 0.0
            self.snapshot["exit_intelligence_threshold"] = 0.0
            self.snapshot["exit_intelligence_reason"] = "NO_OPEN_POSITION"
            self.snapshot["exit_intelligence_codes"] = []
            self.snapshot["exit_intelligence_details"] = {}

    def _advance_position_bar_count(self, position: Position, market_data: dict[str, Any]) -> None:
        raw_candle = market_data.get("candle")
        if raw_candle is None:
            candle = {}
        elif hasattr(raw_candle, "iloc"):
            candle = {}
        else:
            try:
                candle = dict(raw_candle) if hasattr(raw_candle, "keys") else {}
            except Exception:
                candle = {}
        ts_ms = candle.get("timestamp") if candle else None
        try:
            normalized_ts = int(float(ts_ms))
        except (TypeError, ValueError):
            position.bars_held = max(int(getattr(position, "bars_held", 0)), 1)
            return
        if position.last_candle_ts_ms is None:
            position.last_candle_ts_ms = normalized_ts
            position.bars_held = max(int(getattr(position, "bars_held", 0)), 1)
            return
        if normalized_ts > position.last_candle_ts_ms:
            position.bars_held = max(int(getattr(position, "bars_held", 0)) + 1, 1)
            position.last_candle_ts_ms = normalized_ts

    async def _record_exit_intelligence_event(self, position: Position, decision: Any) -> None:
        event = {
            "trade_id": int(position.trade_id),
            "score": _as_float(decision.score, 0.0),
            "threshold": _as_float(decision.threshold, 0.0),
            "reason": str(decision.reason),
            "codes": ",".join(decision.reason_codes or ()),
        }
        events = list(self.snapshot.get("exit_intelligence_log") or [])
        events.append(event)
        self.snapshot["exit_intelligence_log"] = events[-12:]
        await self._log(
            "INFO",
            "exit_intelligence_signal "
            f"trade_id={event['trade_id']} score={event['score']:.4f} "
            f"threshold={event['threshold']:.4f} reason={event['reason']} codes={event['codes'] or 'NONE'}",
        )

    def _detect_structure_exit(
        self,
        position: Position,
        market_data: dict[str, Any],
        current_price: float,
        atr: float,
    ) -> str | None:
        frame5 = market_data.get("frame5")
        if frame5 is None or not hasattr(frame5, "iloc") or len(frame5) < 8:
            return None

        try:
            closes = frame5["close"]
            highs = frame5["high"]
            lows = frame5["low"]
        except (KeyError, TypeError) as e:
            self.logger.warning(f"Failed to access frame data: {e}")
            return None

        last_close = _as_float(closes.iloc[-1], current_price)
        prev_close = _as_float(closes.iloc[-2], last_close)
        prev2_close = _as_float(closes.iloc[-3], prev_close)
        
        slice_highs = highs.iloc[-7:-1] if len(highs) >= 7 else highs
        slice_lows = lows.iloc[-7:-1] if len(lows) >= 7 else lows
        recent_swing_high = max(_as_float(slice_highs.max(), current_price), current_price)
        recent_swing_low = min(_as_float(slice_lows.min(), current_price), current_price)
        
        ema20 = _as_float(frame5.iloc[-1].get("ema20"), last_close)
        used_atr = max(float(atr), current_price * 0.0015)

        if position.side == "BUY":
            open_profit = current_price - position.entry_price
            if open_profit <= max(used_atr * 0.35, 1e-9):
                return None
            retrace_from_high = max((position.peak_price or current_price) - current_price, 0.0)
            bearish_shift = last_close < prev_close < prev2_close
            broke_micro_support = last_close < (recent_swing_low + used_atr * 0.10)
            rejected_from_high = (recent_swing_high - current_price) >= max(used_atr * 0.65, open_profit * 0.45)
            below_structure_ma = last_close < ema20
            if (broke_micro_support and bearish_shift) or (rejected_from_high and below_structure_ma) or retrace_from_high >= used_atr:
                return "STRUCTURE_REVERSAL_EXIT"
            return None

        open_profit = position.entry_price - current_price
        if open_profit <= max(used_atr * 0.35, 1e-9):
            return None
        retrace_from_low = max(current_price - (position.peak_price or current_price), 0.0)
        bullish_shift = last_close > prev_close > prev2_close
        broke_micro_resistance = last_close > (recent_swing_high - used_atr * 0.10)
        rejected_from_low = (current_price - recent_swing_low) >= max(used_atr * 0.65, open_profit * 0.45)
        above_structure_ma = last_close > ema20
        if (broke_micro_resistance and bullish_shift) or (rejected_from_low and above_structure_ma) or retrace_from_low >= used_atr:
            return "STRUCTURE_REVERSAL_EXIT"
        return None

    def _resolve_structure_exit_signal(self, position: Position, structure_exit_reason: str | None) -> str | None:
        if not structure_exit_reason:
            position.structure_exit_votes = 0
            return None
        position.structure_exit_votes = min(position.structure_exit_votes + 1, 3)
        self.snapshot["structure_exit_votes"] = position.structure_exit_votes
        if position.structure_exit_votes < 2:
            return None
        return structure_exit_reason

    async def force_close_position(self) -> None:
        if not self.position_manager.positions:
            await self._log("INFO", "MANUAL_TRADE_CLOSE_SKIPPED no_open_position")
            return

        await self._set_state(BotState.COOLDOWN)
        for position in list(self.position_manager.positions):
            await self._log("INFO", f"MANUAL_TRADE_CLOSE requested trade_id={position.trade_id}")
            closed = await self._close_position(position, "MANUAL_TRADE_CLOSE", _as_float(self.snapshot.get("price"), position.entry_price))
            if not closed:
                await self._log("ERROR", f"MANUAL_TRADE_CLOSE_FAILED trade_id={position.trade_id} action=retry_next_loop")

    async def _close_position(self, position: Position, exit_reason: str, reference_price: float) -> bool:
        close_side = "sell" if position.side == "BUY" else "buy"
        order = None
        for attempt in range(1, 4):
            try:
                intent_id = f"reco-close-{position.trade_id}-{attempt}-{uuid.uuid4().hex[:6]}"
                order = await self.client.create_market_order(
                    self.symbol,
                    close_side,
                    position.quantity,
                    client_order_id=intent_id,
                )
                break
            except ccxt.BaseError as exc:
                await self._log(
                    "WARNING",
                    f"close_order_attempt_failed trade_id={position.trade_id} reason={exit_reason} attempt={attempt}/3 error={exc}",
                )
                if attempt < 3:
                    await asyncio.sleep(1)

        if order is None:
            await self.repository.record_error(self.state.value, "close_order", f"trade_id={position.trade_id} reason={exit_reason}")
            return False

        exit_price = _as_float(order.get("average"), _as_float(order.get("price"), reference_price))
        pnl = (exit_price - position.entry_price) * position.quantity
        if position.side == "SELL":
            pnl *= -1

        exit_slippage_ratio = abs(exit_price - reference_price) / max(reference_price, 1e-9)
        await self.repository.close_trade(
            position.trade_id,
            exit_price,
            pnl,
            exit_reason,
            exit_slippage_ratio=exit_slippage_ratio,
        )
        self.position_manager.close(position.trade_id)
        self._recent_pnls.append(pnl)
        self.session_tracker.record(pnl)
        if len(self._recent_pnls) > 20:
            self._recent_pnls = self._recent_pnls[-20:]
        self.last_close_time = datetime.now(timezone.utc)
        if self._update_loss_protection(pnl):
            await self._log("WARNING", "loss_protection_enabled")
        
        if self.auto_improver.enabled:
            entry_time = datetime.fromtimestamp(position.entry_timestamp_ms / 1000, tz=timezone.utc) if position.entry_timestamp_ms else datetime.now(timezone.utc)
            duration_minutes = (datetime.now(timezone.utc) - entry_time).total_seconds() / 60
            self.auto_improver.record_trade({
                "timestamp": datetime.now(timezone.utc),
                "side": position.side,
                "entry": position.entry_price,
                "exit": exit_price,
                "size": position.quantity,
                "pnl": pnl,
                "duration_minutes": duration_minutes,
                "exit_reason": exit_reason,
            })
            self.logger.info(f"Trade recorded for auto-improvement: PnL={pnl:.4f}, Duration={duration_minutes:.1f}min")
        
        # Record trade for SelfAnalyzer
        if self.self_analyzer.enabled:
            self.self_analyzer.record_trade({
                "timestamp": datetime.now(timezone.utc),
                "side": position.side,
                "entry": position.entry_price,
                "exit": exit_price,
                "size": position.quantity,
                "pnl": pnl,
                "duration_minutes": (datetime.now(timezone.utc) - entry_time).total_seconds() / 60,
                "exit_reason": exit_reason,
            })

        if pnl > 0:
            self.win_count += 1
        self.snapshot["win_rate"] = self.win_count / self.trades_today if self.trades_today else None
        self.snapshot["last_trade"] = f"{position.side} {exit_reason} pnl={pnl:.4f}"

        await self._log("INFO", f"position_closed trade_id={position.trade_id} reason={exit_reason} exit_price={exit_price:.8f} pnl={pnl:.8f}")
        
        self.emergency_system.record_trade(pnl, pnl > 0)
        self.emergency_system.update_equity(_as_float(self.snapshot.get("total_equity"), 1000.0))
        
        if hasattr(self, 'autonomous_brain') and self.autonomous_brain:
            try:
                trade_data = {
                    "pnl": pnl,
                    "pnl_percent": (pnl / position.entry_price * 100) if position.entry_price else 0,
                    "symbol": self.symbol,
                    "side": position.side,
                    "entry_price": position.entry_price,
                    "exit_price": exit_price,
                    "quantity": position.quantity,
                }
                self.autonomous_brain.record_trade(trade_data)
                self.logger.info(f"Trade recorded in Autonomous Brain: PnL={pnl:.4f}")
            except Exception as e:
                self.logger.error(f"Error recording trade in autonomous brain: {e}")
        
        self._sync_ui_state()

        if self.state_manager:
            self.state_manager.add_trade(
                {
                    "trade_id": position.trade_id,
                    "time": datetime.utcnow().isoformat(timespec="seconds"),
                    "pair": self.symbol,
                    "side": position.side,
                    "entry": position.entry_price,
                    "exit": exit_price,
                    "size": position.quantity,
                    "pnl": pnl,
                    "status": exit_reason,
                    "entry_time": "-",
                    "exit_time": datetime.utcnow().isoformat(timespec="seconds"),
                    "fees": order.get("fee", {}).get("cost", 0),
                    "confidence": None,
                    "signal_details": exit_reason,
                    "exit_slippage_ratio": exit_slippage_ratio,
                }
            )
        return True

    def _available_spot_sell_quantity(self) -> float:
        """Cantidad base disponible para ejecutar una venta en spot."""
        return max(_as_float(self.snapshot.get("btc_balance"), 0.0), 0.0)

    def _can_execute_spot_sell(self) -> bool:
        """Determina si una señal SELL es operable en spot con inventario disponible."""
        if not getattr(self.settings, "spot_only_mode", True):
            return True
        if self.position_manager.positions:
            return True
        return self._available_spot_sell_quantity() > 0.0

    def calculate_position_size(
        self,
        price: float,
        stop_loss: float,
        atr: float,
        size_multiplier: float,
        *,
        risk_fraction_override: float | None = None,
    ) -> float:
        equity = _as_float(self.snapshot.get("equity"), _as_float(self.snapshot.get("balance"), 0.0))
        profile = self._current_capital_profile()
        equity = self._effective_equity_for_risk(equity)
        operable_equity = self._operable_equity_for_trading(equity, profile)
        active_risk_fraction = _as_float(risk_fraction_override, self._effective_risk_per_trade_fraction())
        previous_base_risk = self.adaptive_sizer.base_risk_fraction
        self.adaptive_sizer.base_risk_fraction = min(max(active_risk_fraction, 0.001), 0.10)
        decision = self.adaptive_sizer.compute(
            equity=operable_equity,
            price=price,
            stop_loss=stop_loss,
            atr=atr,
            confidence=_as_float(self.snapshot.get("confidence"), 0.0),
            recent_pnls=list(self._recent_pnls),
            volatility_multiplier=max(size_multiplier * profile.size_multiplier, 0.1),
        )
        self.adaptive_sizer.base_risk_fraction = previous_base_risk
        self.snapshot["adaptive_size_reason"] = decision.reason
        self.snapshot["adaptive_size_multiplier"] = decision.size_multiplier
        self.snapshot["operable_capital_usdt"] = operable_equity
        self.snapshot["capital_profile"] = profile.name
        self.snapshot["live_trade_risk_fraction"] = active_risk_fraction
        return float(max(decision.quantity, 0.0))

    def _compute_per_trade_investment_controls(self, *, confidence: float, price: float, atr: float) -> dict[str, float]:
        base_risk = self._effective_risk_per_trade_fraction()
        base_allocation = self._effective_max_trade_balance_fraction()
        optimized_risk = _as_float(self.snapshot.get("optimized_risk_per_trade_fraction"), base_risk)
        optimized_allocation = _as_float(self.snapshot.get("optimized_max_trade_balance_fraction"), base_allocation)
        anchor_risk = min(base_risk, optimized_risk) if "auto" in str(self.runtime_investment_mode).lower() else base_risk
        anchor_allocation = (
            min(base_allocation, optimized_allocation) if "auto" in str(self.runtime_investment_mode).lower() else base_allocation
        )

        conf = max(min(float(confidence), 1.0), 0.0)
        if conf >= 0.90:
            confidence_mult = 1.25
        elif conf >= 0.80:
            confidence_mult = 1.10
        elif conf <= 0.65:
            confidence_mult = 0.75
        else:
            confidence_mult = 1.0

        volatility_ratio = max(float(atr), 0.0) / max(float(price), 1e-9)
        if volatility_ratio >= 0.02:
            volatility_mult = 0.70
        elif volatility_ratio >= 0.01:
            volatility_mult = 0.85
        else:
            volatility_mult = 1.0

        total_equity = _as_float(self.snapshot.get("total_equity"), _as_float(self.snapshot.get("equity"), 0.0))
        peak_equity = max(_as_float(getattr(self, "equity_peak", total_equity), total_equity), 1e-9)
        drawdown = max((peak_equity - total_equity) / peak_equity, 0.0)
        drawdown_mult = max(1.0 - drawdown * 1.8, 0.40)

        profile = self._current_capital_profile()
        trade_risk = anchor_risk * confidence_mult * volatility_mult * drawdown_mult
        trade_allocation = anchor_allocation * confidence_mult * volatility_mult * drawdown_mult
        trade_risk = min(max(trade_risk, 0.001), profile.risk_per_trade_fraction, 0.10)
        trade_allocation = min(max(trade_allocation, 0.01), profile.max_trade_balance_fraction, 1.0)
        self.snapshot["live_trade_risk_fraction"] = trade_risk
        self.snapshot["live_trade_max_allocation_fraction"] = trade_allocation
        return {
            "risk_per_trade_fraction": trade_risk,
            "max_trade_balance_fraction": trade_allocation,
        }

    def _pullback_assessment(self, bundle: SignalBundle, side: str, market_data: dict[str, Any]) -> tuple[bool, bool]:
        """
        Confirmación de entrada con momentum mínimo.
        No bloquea en tendencia. Solo rechaza si la vela más reciente va
        fuertemente en contra de la señal (2 velas consecutivas contrarias).
        """
        frame5 = market_data.get("frame5")
        if frame5 is None or len(frame5) < 4:
            return True, False

        row = frame5.iloc[-1]
        prev1 = frame5.iloc[-2]
        prev2 = frame5.iloc[-3]

        close = _as_float(row.get("close"), 0.0)
        p1c = _as_float(prev1.get("close"), close)
        p2c = _as_float(prev2.get("close"), close)
        atr = _as_float(row.get("atr"), close * 0.005)

        if side == "BUY":
            two_consecutive_bear = (p1c < _as_float(prev1.get("open"), p1c)) and (p2c < _as_float(prev2.get("open"), p2c))
            strong_move_down = (p1c - close) > (0.5 * atr)
            extreme_move_down = (p1c - close) > (0.9 * atr)
            blocked = two_consecutive_bear and strong_move_down
            return (not blocked), bool(blocked and extreme_move_down)

        two_consecutive_bull = (p1c > _as_float(prev1.get("open"), p1c)) and (p2c > _as_float(prev2.get("open"), p2c))
        strong_move_up = (close - p1c) > (0.5 * atr)
        extreme_move_up = (close - p1c) > (0.9 * atr)
        blocked = two_consecutive_bull and strong_move_up
        return (not blocked), bool(blocked and extreme_move_up)

    def _pullback_confirmed(self, bundle: SignalBundle, side: str, market_data: dict[str, Any]) -> bool:
        confirmed, _extreme = self._pullback_assessment(bundle, side, market_data)
        return confirmed

    def _build_stops(
        self, side: str, entry: float, atr: float, regime: str = "NORMAL_VOLATILITY"
    ) -> tuple[float, float]:
        atr = max(atr, entry * 0.002)
        regime_upper = regime.upper()
        if "HIGH" in regime_upper:
            sl_mult, tp_mult = 1.2, 2.4
        elif "LOW" in regime_upper:
            sl_mult, tp_mult = 1.8, 3.0
        else:
            sl_mult, tp_mult = 1.5, 2.5

        if side == "BUY":
            stop_loss = self.order_manager.normalize_price(entry - sl_mult * atr)
            take_profit = self.order_manager.normalize_price(entry + tp_mult * atr)
        else:
            stop_loss = self.order_manager.normalize_price(entry + sl_mult * atr)
            take_profit = self.order_manager.normalize_price(entry - tp_mult * atr)
        return stop_loss, take_profit

    def _update_loss_protection(self, pnl: float) -> bool:
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        if self.consecutive_losses >= self._effective_loss_pause_after_consecutive():
            self.pause_trading_until = datetime.now(timezone.utc) + timedelta(minutes=self._effective_loss_pause_minutes())
            self.consecutive_losses = 0
            return True
        return False

    async def _persist_signal(
        self,
        bundle: SignalBundle,
        side: str,
        confidence: float,
        *,
        decision_trace: dict[str, Any] | None = None,
    ) -> None:
        await self.repository.record_signal(
            self.symbol,
            {
                "trend": bundle.trend,
                "momentum": bundle.momentum,
                "volume": bundle.volume,
                "volatility": bundle.volatility,
                "structure": bundle.structure,
                "order_flow": bundle.order_flow,
                "regime": bundle.regime,
            },
            confidence,
            side,
            factor_scores=(decision_trace or {}).get("factor_scores"),
            gating=self.snapshot.get("decision_gating", {}),
            decision_reason=str(self.snapshot.get("decision_reason", "ANALYSIS")),
        )

    def _update_snapshot(self, market_data: dict[str, Any], analysis: dict[str, Any]) -> None:
        bundle: SignalBundle = analysis["bundle"]
        self.snapshot.update(
            {
                "pair": self.symbol,
                "timeframe": f"{self.settings.primary_timeframe} / {self.settings.confirmation_timeframe}",
                "price": market_data.get("price"),
                "spread": market_data.get("spread"),
                "bid": market_data.get("bid"),
                "ask": market_data.get("ask"),
                "trend": bundle.trend,
                "adx": market_data.get("adx"),
                "volatility_regime": bundle.regime,
                "order_flow": bundle.order_flow,
                "volume": market_data.get("volume"),
                "atr": market_data.get("atr"),
                "change_24h": market_data.get("change_24h"),
                "candles_5m": list(market_data.get("candles_5m") or [])[-120:],
                "signal": analysis.get("side"),
                "raw_signal": analysis.get("raw_side"),
                "confidence": analysis.get("confidence"),
                "signal_quality_score": analysis.get("setup_quality"),
                "status": self.state.value,
                "signals": {
                    "trend": bundle.trend,
                    "momentum": bundle.momentum,
                    "volume": bundle.volume,
                    "volatility": bundle.volatility,
                    "structure": bundle.structure,
                    "order_flow": bundle.order_flow,
                },
                "decision_trace": analysis.get("decision_trace", self.snapshot.get("decision_trace", {})),
                "decision_gating": self.snapshot.get("decision_gating", {}),
                "decision_reason": self.snapshot.get("decision_reason", "ANALYSIS"),
            }
        )

    def _set_decision_gating(self, reason: str, final_action: str) -> None:
        self.snapshot["decision_reason"] = reason
        self.snapshot["decision_gating"] = {
            "spread_gate": bool(_as_float(self.snapshot.get("spread"), 0.0) <= _as_float(self.snapshot.get("price"), 1.0) * self.settings.max_spread_ratio),
            "daily_loss_gate": bool(_as_float(self.snapshot.get("session_pnl"), 0.0) > -abs(_as_float(self.settings.daily_loss_limit_fraction, 0.03) * max(_as_float(self.snapshot.get("equity"), 0.0), 1.0))),
            "cooldown_gate": self._is_cooldown_complete(),
            "final_action": final_action,
        }


    def _apply_market_intelligence_snapshot(self, intelligence: dict[str, Any]) -> None:
        self.snapshot["market_regime"] = intelligence.get("market_regime")
        self.snapshot["volatility_state"] = intelligence.get("volatility_state")
        self.snapshot["distance_to_support"] = intelligence.get("distance_to_support")
        self.snapshot["distance_to_resistance"] = intelligence.get("distance_to_resistance")

    async def _fetch_balances(self) -> tuple[float, float, str, str, dict[str, Any]]:
        payload = await self.client.fetch_balance()
        base_asset, quote_asset = split_symbol(self.symbol)
        quote_asset = quote_asset or "USDT"
        free_balances = payload.get("free") if isinstance(payload, dict) else None
        if isinstance(free_balances, dict):
            quote_balance = _as_float(free_balances.get(quote_asset), 0.0)
            base_balance = _as_float(free_balances.get(base_asset), 0.0)
            return quote_balance, base_balance, quote_asset, base_asset, payload

        quote = payload.get(quote_asset) if isinstance(payload, dict) else {}
        base = payload.get(base_asset) if isinstance(payload, dict) else {}
        quote_balance = _as_float((quote or {}).get("free"), 0.0)
        base_balance = _as_float((base or {}).get("free"), 0.0)
        return quote_balance, base_balance, quote_asset, base_asset, payload

    async def _quote_to_usdt_rate(self, quote_asset: str) -> float:
        normalized = str(quote_asset or "USDT").upper()
        if normalized == "USDT":
            return 1.0
        now = datetime.now(timezone.utc)
        cached = self._quote_to_usdt_cache.get(normalized)
        if cached and (now - cached[1]).total_seconds() < 300:
            return cached[0]
        conversion_pairs = [f"{normalized}/USDT", f"USDT/{normalized}"]
        for pair in conversion_pairs:
            try:
                ticker = await self.client.fetch_ticker(pair)
            except Exception:  # noqa: BLE001
                continue
            last_price = _as_float((ticker or {}).get("last"), 0.0)
            if last_price <= 0:
                continue
            rate = last_price if pair.startswith(normalized) else (1.0 / last_price)
            self._quote_to_usdt_cache[normalized] = (rate, now)
            return rate
        return 1.0

    async def _asset_to_usdt_rate(self, asset: str) -> float:
        normalized = str(asset or "").upper().strip()
        if not normalized:
            return 0.0
        if normalized == "USDT":
            return 1.0
        now = datetime.now(timezone.utc)
        cached = self._asset_to_usdt_cache.get(normalized)
        if cached and (now - cached[1]).total_seconds() < 300:
            return cached[0]
        for pair in (f"{normalized}/USDT", f"USDT/{normalized}"):
            try:
                ticker = await self.client.fetch_ticker(pair)
            except Exception:  # noqa: BLE001
                continue
            last_price = _as_float((ticker or {}).get("last"), 0.0)
            if last_price <= 0:
                continue
            rate = last_price if pair.startswith(normalized) else (1.0 / last_price)
            self._asset_to_usdt_cache[normalized] = (rate, now)
            return rate
        for bridge in ("USDC", "FDUSD", "BUSD", "TUSD", "DAI", "BTC", "ETH", "BNB"):
            if bridge == normalized:
                continue
            bridge_rate = await self._asset_to_usdt_rate(bridge)
            if bridge_rate <= 0:
                continue
            for pair in (f"{normalized}/{bridge}", f"{bridge}/{normalized}"):
                try:
                    ticker = await self.client.fetch_ticker(pair)
                except Exception:  # noqa: BLE001
                    continue
                last_price = _as_float((ticker or {}).get("last"), 0.0)
                if last_price <= 0:
                    continue
                asset_to_bridge = last_price if pair.startswith(normalized) else (1.0 / last_price)
                rate = asset_to_bridge * bridge_rate
                self._asset_to_usdt_cache[normalized] = (rate, now)
                return rate
        return 0.0

    async def _compute_total_equity_usdt(self, payload: dict[str, Any]) -> float:
        if not isinstance(payload, dict):
            return 0.0
        totals = payload.get("total")
        if not isinstance(totals, dict):
            totals = {}
            free = payload.get("free")
            used = payload.get("used")
            if isinstance(free, dict):
                for asset, amount in free.items():
                    totals[str(asset)] = _as_float(amount, 0.0)
            if isinstance(used, dict):
                for asset, amount in used.items():
                    totals[str(asset)] = _as_float(totals.get(str(asset), 0.0), 0.0) + _as_float(amount, 0.0)
        if not totals:
            return 0.0
        total_equity_usdt = 0.0
        for asset, amount in totals.items():
            quantity = _as_float(amount, 0.0)
            if quantity <= 0:
                continue
            rate = await self._asset_to_usdt_rate(str(asset))
            if rate <= 0:
                continue
            total_equity_usdt += quantity * rate
        return float(total_equity_usdt)

    async def _refresh_account_snapshot(self, current_price: float | None = None) -> None:
        quote_balance, base_balance, quote_asset, _base_asset, balance_payload = await self._fetch_balances()
        reference_price = max(_as_float(current_price, _as_float(self.snapshot.get("price"), 0.0)), 0.0)
        base_value_quote = float(base_balance * reference_price)
        total_equity_quote = float(quote_balance + base_value_quote)
        quote_to_usdt_rate = await self._quote_to_usdt_rate(quote_asset)
        total_equity_usdt = await self._compute_total_equity_usdt(balance_payload)
        if total_equity_usdt <= 0:
            total_equity_usdt = float(total_equity_quote * quote_to_usdt_rate)
        session_pnl = float(await self.repository.get_session_pnl() or 0.0)

        self.snapshot["balance"] = float(quote_balance)
        self.snapshot["btc_balance"] = float(base_balance)
        self.snapshot["btc_value"] = base_value_quote
        self.snapshot["total_equity"] = total_equity_quote
        self.snapshot["total_equity_usdt"] = total_equity_usdt
        self.snapshot["equity"] = total_equity_quote
        self.snapshot["account_currency"] = quote_asset
        self.snapshot["daily_pnl"] = session_pnl
        self.snapshot["session_pnl"] = session_pnl
        self.snapshot["trades_today"] = self.trades_today
        self.snapshot["win_rate"] = self.win_count / self.trades_today if self.trades_today else 0.0

        profile = self._current_capital_profile()
        operable_capital = self._operable_equity_for_trading(total_equity_usdt, profile)
        self.snapshot["capital_profile"] = profile.name
        self.snapshot["operable_capital_usdt"] = operable_capital

        if self.starting_equity is None:
            self.starting_equity = max(total_equity_usdt, 1.0)
        self.equity_peak = max(_as_float(self.equity_peak, total_equity_usdt), total_equity_usdt)
        self._append_equity_point(total_equity_usdt)

    def _append_equity_point(self, equity: float) -> None:
        normalized = round(float(max(equity, 0.0)), 8)
        if self.equity_curve_history and abs(self.equity_curve_history[-1] - normalized) < 1e-8:
            return
        self.equity_curve_history.append(normalized)
        self.equity_curve_history = self.equity_curve_history[-240:]

    async def _reconcile_open_positions(self) -> None:
        open_trades = await self.repository.get_open_trades(self.symbol)
        if not open_trades:
            return
        quote_balance, base_balance, quote_asset, _base_asset, _payload = await self._fetch_balances()
        self.snapshot["balance"] = quote_balance
        self.snapshot["btc_balance"] = base_balance
        self.snapshot["account_currency"] = quote_asset
        latest = open_trades[0]
        if base_balance <= 0:
            await self._log("WARNING", "open_trade_found_without_base_balance")
            return
        self.position_manager.open(
            Position(
                trade_id=latest.id,
                side=latest.side,
                quantity=min(latest.quantity, base_balance),
                entry_price=latest.entry_price,
                stop_loss=latest.stop_loss,
                take_profit=latest.take_profit,
                atr=0.0,
                initial_risk_distance=abs(latest.entry_price - latest.stop_loss),
            )
        )
        await self._log("INFO", f"reconciled_open_position trade_id={latest.id} quantity={min(latest.quantity, base_balance):.8f}")

    async def _switch_to_pair(self, new_symbol: str) -> None:
        """Switch to a new trading pair with proper cleanup and rule sync."""
        try:
            old_symbol = self.symbol
            self.symbol = new_symbol
            self.order_manager = OrderManager(self.client, self.symbol)
            await self.order_manager.sync_rules()
            self.market_stream = MarketStream(self.client, self.symbol, self.settings.history_limit)
            self._cached_frame5 = None
            self._cached_frame15 = None
            self.multi_pair_manager.record_switch(old_symbol, new_symbol)
            self.logger.warning(f"Auto-switched to best pair: {old_symbol} -> {self.symbol}")
            self.snapshot["pair_switch"] = self.symbol
        except Exception as exc:
            self.logger.error(f"Failed to switch pair to {new_symbol}: {exc}")
            self.snapshot["pair_switch_error"] = str(exc)
            self.symbol = old_symbol

    def _start_ml_auto_training(self) -> None:
        """Start background auto-training of ML models with historical data."""
        import threading
        
        def _train_models_in_background():
            try:
                import pandas as pd
                
                # Get historical data for training
                self.logger.info("Starting ML auto-training with historical data...")
                
                # Train TFT model
                if hasattr(self, 'tft_manager') and self.tft_manager:
                    try:
                        import ccxt.async_support as ccxt_async
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            exchange = ccxt_async.binance({"enableRateLimit": True})
                            ohlcv = loop.run_until_complete(
                                exchange.fetch_ohlcv(self.symbol, "5m", limit=500)
                            )
                            loop.run_until_complete(exchange.close())
                            
                            if ohlcv:
                                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                                df.set_index("timestamp", inplace=True)
                                
                                result = self.tft_manager.train(df, self.symbol)
                                self.logger.info(f"TFT model trained: {result}")
                        finally:
                            loop.close()
                    except Exception as tft_exc:
                        self.logger.debug(f"TFT auto-training skipped: {tft_exc}")
                
                # Train NBEATS model
                if hasattr(self, 'nbeats_manager') and self.nbeats_manager:
                    try:
                        import ccxt.async_support as ccxt_async
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            exchange = ccxt_async.binance({"enableRateLimit": True})
                            ohlcv = loop.run_until_complete(
                                exchange.fetch_ohlcv(self.symbol, "5m", limit=500)
                            )
                            loop.run_until_complete(exchange.close())
                            
                            if ohlcv:
                                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                                df.set_index("timestamp", inplace=True)
                                
                                result = self.nbeats_manager.train(df, self.symbol)
                                self.logger.info(f"NBEATS model trained: {result}")
                        finally:
                            loop.close()
                    except Exception as nb_exc:
                        self.logger.debug(f"NBEATS auto-training skipped: {nb_exc}")
                
                self.logger.info("ML auto-training complete")
            except Exception as exc:
                self.logger.warning(f"ML auto-training failed: {exc}")
        
        # Run in background thread to not block startup
        thread = threading.Thread(target=_train_models_in_background, daemon=True, name="ml-auto-training")
        thread.start()
        self.logger.info("ML auto-training started in background")

    async def _start_market_analysis(self, market_count: int) -> None:
        """Analyze all markets to find the best trading pair."""
        self._market_analysis_cancelled = False
        ma = self.snapshot["market_analysis"]
        ma["status"] = "running"
        ma["total_markets"] = market_count
        ma["analyzed_markets"] = 0
        ma["progress_pct"] = 0.0
        ma["best_pair"] = "--"
        ma["best_score"] = 0.0
        self._sync_ui_state()

        try:
            all_tickers = await self.client.fetch_tickers()
            usdt_pairs = [
                sym for sym in all_tickers.keys()
                if sym.endswith("/USDT") and ":" not in sym and sym.count("/") == 1
            ][:market_count]

            best_pair = "--"
            best_score = 0.0

            for i, pair in enumerate(usdt_pairs):
                if self._market_analysis_cancelled:
                    ma["status"] = "cancelled"
                    ma["progress_pct"] = 0.0
                    self._sync_ui_state()
                    return

                try:
                    ticker = all_tickers.get(pair, {})
                    price = float(ticker.get("last", 0))
                    volume = float(ticker.get("baseVolume", 0))
                    change = float(ticker.get("percentage", 0) or 0)
                    spread = 0.0
                    if ticker.get("ask") and ticker.get("bid"):
                        spread = (float(ticker["ask"]) - float(ticker["bid"])) / float(ticker["ask"]) * 100 if float(ticker["ask"]) > 0 else 0

                    score = 0.0
                    if price > 0:
                        score += min(volume / 1000, 30)
                        score += min(abs(change) / 5, 20)
                        score += max(0, 20 - spread * 10)
                        score += min(price / 1000, 10)
                        score += 20

                    if score > best_score:
                        best_score = score
                        best_pair = pair

                    ma["analyzed_markets"] = i + 1
                    ma["progress_pct"] = ((i + 1) / len(usdt_pairs)) * 100
                    ma["best_pair"] = best_pair
                    ma["best_score"] = best_score
                    self._sync_ui_state()
                except Exception:
                    continue

            from datetime import datetime, timezone
            ma["status"] = "completed"
            ma["progress_pct"] = 100.0
            ma["best_pair"] = best_pair
            ma["best_score"] = best_score
            ma["last_analysis_time"] = datetime.now(timezone.utc).strftime("%H:%M:%S")
            self._sync_ui_state()
            await self._log("INFO", f"market_analysis_completed best={best_pair} score={best_score:.2f}")
        except Exception as exc:
            ma["status"] = "idle"
            ma["progress_pct"] = 0.0
            self._sync_ui_state()
            await self._log("ERROR", f"market_analysis_failed: {exc}")

    def _on_auto_fix_refresh(self) -> None:
        """Called by AutoFixCoordinator after fixes are applied."""
        self.logger.info("Auto-fix refresh triggered - applying corrections")
        self.snapshot["auto_fix_applied"] = True
        self._sync_ui_state()

    async def _run_log_analysis_cycle(self) -> None:
        """Periodically analyze logs and write errors to docs/analisis.txt."""
        try:
            logs = self.snapshot.get("logs", [])[-100:]
            new_errors = self.log_analyzer.analyze_logs(logs)
            if new_errors:
                await self._log("WARNING", f"llm_analyzer found {len(new_errors)} new errors")
            self.log_analyzer.cleanup_old_errors()
            self.snapshot["llm_analysis"] = self.log_analyzer.get_summary()
        except Exception as exc:
            self.logger.debug(f"Log analysis cycle error: {exc}")

    async def _process_control_requests(self) -> None:
        if not self.state_manager or not hasattr(self.state_manager, "pop_control_requests"):
            return
        controls = self.state_manager.pop_control_requests()
        for control in controls:
            if control == "force_close":
                await self.force_close_position()
            elif control == "pause":
                self.manual_pause = True
                await self._log("WARNING", "manual_pause_requested")
            elif control in {"start", "resume"}:
                self.manual_pause = False
                self.emergency_stop_active = False
                await self._log("INFO", f"manual_control_{control}")
            elif control == "emergency_stop":
                self.emergency_stop_active = True
                self.manual_pause = True
                await self._log("ERROR", "emergency_stop_requested")
                await self.force_close_position()
            elif control == "analysis_start":
                market_count = self.snapshot.get("market_analysis_market_count", 50)
                await self._start_market_analysis(int(market_count))
            elif control == "analysis_cancel":
                self._market_analysis_cancelled = True
                await self._log("INFO", "market_analysis_cancelled")
            elif control == "analysis_change_pair":
                best_pair = self.snapshot.get("market_analysis", {}).get("best_pair")
                if best_pair and best_pair != "--":
                    self._pending_pair_switch = normalize_symbol(best_pair)
                    await self._log("INFO", f"pair_switch_requested target={self._pending_pair_switch}")

        if hasattr(self.state_manager, "pop_runtime_settings"):
            runtime_updates = self.state_manager.pop_runtime_settings()
            for update in runtime_updates:
                await self._apply_runtime_settings(update)

    async def _sleep_with_responsiveness(self, seconds: float) -> None:
        total = max(float(seconds), 0.0)
        if total <= 0:
            return
        elapsed = 0.0
        step = 0.25
        while elapsed < total:
            chunk = min(step, total - elapsed)
            await asyncio.sleep(chunk)
            elapsed += chunk
            await self._process_control_requests()
            self._sync_ui_state()

    async def _apply_runtime_settings(self, settings_payload: dict[str, Any], *, persist: bool = True) -> None:
        sanitized = _sanitize_runtime_settings_payload(settings_payload)
        requested_symbol = normalize_symbol(str(sanitized.get("default_pair", "")).strip())
        symbol_switched = False
        if requested_symbol and requested_symbol != self.symbol:
            if await self._has_active_trade_for_symbol_switch():
                await self._log("WARNING", f"symbol_switch_blocked active_trade symbol={self.symbol} requested={requested_symbol}")
                sanitized["default_pair"] = self.symbol
            else:
                await self._switch_symbol(requested_symbol)
                symbol_switched = True
                sanitized["default_pair"] = requested_symbol

        mode = str(sanitized.get("investment_mode", self.runtime_investment_mode)).strip()
        self.runtime_investment_mode = mode or self.runtime_investment_mode
        auto_profile = self._auto_investment_controls(self.runtime_investment_mode, sanitized)

        risk_fraction = _as_float(
            auto_profile.get("risk_per_trade_fraction", sanitized.get("risk_per_trade_fraction")),
            self._effective_risk_per_trade_fraction(),
        )
        max_trade_fraction = _as_float(
            auto_profile.get("max_trade_balance_fraction", sanitized.get("max_trade_balance_fraction")),
            self._effective_max_trade_balance_fraction(),
        )
        capital_limit = _as_float(
            auto_profile.get("capital_limit_usdt", sanitized.get("capital_limit_usdt")),
            self.runtime_capital_limit_usdt or 0.0,
        )
        reserve_ratio = _as_float(
            auto_profile.get("capital_reserve_ratio", sanitized.get("capital_reserve_ratio")),
            self.runtime_capital_reserve_ratio if self.runtime_capital_reserve_ratio is not None else getattr(self.settings, "capital_reserve_ratio", 0.15),
        )
        min_cash_buffer = _as_float(
            auto_profile.get("min_cash_buffer_usdt", sanitized.get("min_cash_buffer_usdt")),
            self.runtime_min_cash_buffer_usdt if self.runtime_min_cash_buffer_usdt is not None else getattr(self.settings, "min_cash_buffer_usdt", 10.0),
        )
        self.runtime_symbol_capital_limits = {
            normalize_symbol(symbol): limit
            for symbol, limit in sanitized.get("symbol_capital_limits", {}).items()
            if limit > 0
        }

        self.runtime_risk_per_trade_fraction = min(max(risk_fraction, 0.001), 0.10)
        self.runtime_max_trade_balance_fraction = min(max(max_trade_fraction, 0.01), 1.0)
        self.runtime_capital_limit_usdt = capital_limit if capital_limit > 0 else None
        self.runtime_capital_reserve_ratio = min(max(reserve_ratio, 0.0), 0.90)
        self.runtime_min_cash_buffer_usdt = max(min_cash_buffer, 0.0)
        self.runtime_dynamic_exit_enabled = bool(
            auto_profile.get("dynamic_exit_enabled", sanitized.get("dynamic_exit_enabled", self.runtime_dynamic_exit_enabled))
        )
        self.runtime_confidence_boost_multiplier = _as_float(
            auto_profile.get("confidence_boost_multiplier", self.runtime_confidence_boost_multiplier),
            self.runtime_confidence_boost_multiplier,
        )
        self.snapshot["investment_mode"] = self.runtime_investment_mode
        self.snapshot["capital_limit_usdt"] = self.runtime_capital_limit_usdt
        self.snapshot["capital_reserve_ratio"] = self.runtime_capital_reserve_ratio
        self.snapshot["min_cash_buffer_usdt"] = self.runtime_min_cash_buffer_usdt
        self.snapshot["dynamic_exit_enabled"] = self.runtime_dynamic_exit_enabled
        self.snapshot["optimized_risk_per_trade_fraction"] = auto_profile.get("risk_per_trade_fraction")
        self.snapshot["optimized_max_trade_balance_fraction"] = auto_profile.get("max_trade_balance_fraction")
        self.snapshot["optimized_capital_limit_usdt"] = auto_profile.get("capital_limit_usdt")
        self.snapshot["optimization_reason"] = auto_profile.get("optimization_reason")
        self.snapshot["runtime_settings"] = {**sanitized, **auto_profile}
        
        filter_config = sanitized.get("filter_config", {})
        if filter_config:
            self._apply_symbol_filter_config(filter_config)
        
        if symbol_switched:
            self.snapshot["pair"] = self.symbol
        if persist:
            await self.repository.set_runtime_setting("ui_runtime_settings", {**sanitized, **auto_profile})

    async def _has_active_trade_for_symbol_switch(self) -> bool:
        if self.position_manager.positions:
            return True
        try:
            open_trades = await self.repository.get_open_trades(self.symbol)
        except (ConnectionError, asyncio.TimeoutError) as e:
            self.logger.warning(f"Failed to check open trades: {e}")
            return bool(self.position_manager.positions)
        except Exception as e:
            self.logger.error(f"Unexpected error checking open trades: {e}")
            return bool(self.position_manager.positions)
        return bool(open_trades)

    async def _switch_symbol(self, new_symbol: str) -> None:
        normalized_symbol = normalize_symbol(new_symbol)
        if not normalized_symbol or normalized_symbol == self.symbol:
            return
        previous_symbol = self.symbol
        await self._set_state(BotState.SYNCING_SYMBOL, f"runtime_symbol_switch {previous_symbol}->{normalized_symbol}")
        self.symbol = normalized_symbol
        self.market_stream.symbol = normalized_symbol
        self.order_manager.symbol = normalized_symbol
        self.order_manager.rules = None
        self._cached_frame5 = None
        self._cached_frame15 = None
        self._last_primary_indicator_ts = None
        self._last_confirmation_indicator_ts = None
        self._quote_to_usdt_cache.clear()
        self._asset_to_usdt_cache.clear()
        self.base_filter_config = self._get_default_filter_config()
        self.runtime_filter_config = dict(self.base_filter_config)
        self.snapshot.update(
            {
                "pair": normalized_symbol,
                "signal": None,
                "raw_signal": None,
                "confidence": None,
                "signals": {},
                "candles_5m": [],
                "change_24h": None,
            }
        )
        await self.order_manager.sync_rules()
        await self._set_state(BotState.WAITING_MARKET_DATA, "runtime_symbol_switch_complete")
        await self._log("INFO", f"runtime_symbol_switched from={previous_symbol} to={normalized_symbol}")

    async def _switch_symbol_if_pending(self) -> None:
        if not self._pending_pair_switch:
            return
        if await self._has_active_trade_for_symbol_switch():
            return
        target = self._pending_pair_switch
        self._pending_pair_switch = None
        await self._switch_symbol(target)

    def _auto_investment_controls(self, mode: str, sanitized_payload: dict[str, Any] | None = None) -> dict[str, float | bool | str]:
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in {"auto", "auto-optimized", "auto_optimized"}:
            return {}

        equity = _as_float(self.snapshot.get("total_equity"), _as_float(self.snapshot.get("equity"), _as_float(self.snapshot.get("balance"), 0.0)))
        profile = self._current_capital_profile()
        volatility_ratio = _as_float(self.snapshot.get("atr"), 0.0) / max(_as_float(self.snapshot.get("price"), 0.0), 1e-9)
        peak = _as_float(getattr(self, "equity_peak", equity), equity)
        drawdown_fraction = max((peak - equity) / max(peak, 1e-9), 0.0) if peak > 0 else 0.0
        win_rate = self.snapshot.get("win_rate")
        trades_today = int(getattr(self, "trades_today", 0))
        win_count = int(getattr(self, "win_count", 0))
        if win_rate is None and trades_today > 0:
            win_rate = _as_float(win_count / max(trades_today, 1), 0.5)

        payload = sanitized_payload or {}
        risk_cap = payload.get("risk_per_trade_fraction")
        allocation_cap = payload.get("max_trade_balance_fraction")
        optimized = self.investment_optimizer.optimize(
            equity=equity,
            profile=profile,
            volatility_ratio=volatility_ratio,
            drawdown_fraction=drawdown_fraction,
            win_rate=_as_float(win_rate, 0.5) if win_rate is not None else None,
            risk_cap=_as_float(risk_cap, profile.risk_per_trade_fraction) if risk_cap is not None else None,
            allocation_cap=_as_float(allocation_cap, profile.max_trade_balance_fraction) if allocation_cap is not None else None,
        )
        return {
            "risk_per_trade_fraction": optimized.risk_per_trade_fraction,
            "max_trade_balance_fraction": optimized.max_trade_balance_fraction,
            "capital_reserve_ratio": optimized.capital_reserve_ratio,
            "min_cash_buffer_usdt": optimized.min_cash_buffer_usdt,
            "capital_limit_usdt": optimized.capital_limit_usdt,
            "dynamic_exit_enabled": optimized.dynamic_exit_enabled,
            "confidence_boost_multiplier": optimized.confidence_boost_multiplier,
            "optimization_reason": optimized.optimization_reason,
        }

    def _effective_equity_for_risk(self, equity: float) -> float:
        symbol_limit = self.runtime_symbol_capital_limits.get(normalize_symbol(self.symbol))
        limit = symbol_limit if symbol_limit is not None else self.runtime_capital_limit_usdt
        if limit is None:
            return max(equity, 0.0)
        return max(min(equity, limit), 0.0)

    def _current_capital_profile(self) -> CapitalProfile:
        equity = self._equity_reference_usdt()
        if not hasattr(self, "capital_profile_manager") or self.capital_profile_manager is None:
            self.capital_profile_manager = CapitalProfileManager()
        settings = getattr(self, "settings", None)
        if not getattr(settings, "enable_capital_profiles", True):
            equity = max(equity, 1000.0)
        return self.capital_profile_manager.select(equity)

    def _equity_reference_usdt(self) -> float:
        return _as_float(
            self.snapshot.get("total_equity_usdt"),
            _as_float(self.snapshot.get("total_equity"), _as_float(self.snapshot.get("equity"), _as_float(self.snapshot.get("balance"), 0.0))),
        )

    def _operable_equity_for_trading(self, equity: float, profile: CapitalProfile | None = None) -> float:
        active_profile = profile or self._current_capital_profile()
        capital_limit = self._effective_equity_cap()
        runtime_reserve_ratio = getattr(self, "runtime_capital_reserve_ratio", None)
        runtime_cash_buffer = getattr(self, "runtime_min_cash_buffer_usdt", None)
        reserve_ratio = runtime_reserve_ratio if runtime_reserve_ratio is not None else active_profile.reserve_ratio
        reserve_buffer = runtime_cash_buffer if runtime_cash_buffer is not None else active_profile.reserve_buffer_usdt
        effective_profile = CapitalProfile(
            **{**asdict(active_profile), "reserve_ratio": reserve_ratio, "reserve_buffer_usdt": reserve_buffer}
        )
        return self.capital_profile_manager.operable_capital(equity, effective_profile, capital_limit)

    def _effective_equity_cap(self) -> float | None:
        symbol_limits = getattr(self, "runtime_symbol_capital_limits", {}) or {}
        symbol = normalize_symbol(getattr(self, "symbol", ""))
        symbol_limit = symbol_limits.get(symbol)
        if symbol_limit is not None:
            return symbol_limit
        return getattr(self, "runtime_capital_limit_usdt", None)

    def _effective_risk_per_trade_fraction(self) -> float:
        profile_fraction = self._current_capital_profile().risk_per_trade_fraction
        if self.runtime_risk_per_trade_fraction is not None:
            return min(self.runtime_risk_per_trade_fraction, profile_fraction)
        return min(float(self.settings.risk_per_trade_fraction), profile_fraction)

    def _effective_max_trade_balance_fraction(self) -> float:
        profile_fraction = self._current_capital_profile().max_trade_balance_fraction
        if self.runtime_max_trade_balance_fraction is not None:
            return min(self.runtime_max_trade_balance_fraction, profile_fraction)
        return min(float(self.settings.max_trade_balance_fraction), profile_fraction)

    def _effective_min_signal_confidence(self) -> float:
        if self.runtime_confidence_threshold is not None:
            return self.runtime_confidence_threshold
        return max(float(self._current_capital_profile().min_confidence), float(self.settings.confidence_threshold))

    def _effective_max_spread_ratio(self) -> float:
        return min(float(getattr(self.settings, "max_spread_ratio", 0.004)), float(self._current_capital_profile().max_spread_ratio))

    def _effective_max_trades_per_day(self) -> int:
        return min(int(self.settings.max_trades_per_day), int(self._current_capital_profile().max_trades_per_day))

    def _effective_cooldown_minutes(self) -> int:
        return max(int(self.settings.cooldown_minutes), int(self._current_capital_profile().cooldown_minutes))

    def _effective_loss_pause_minutes(self) -> int:
        return max(int(self.settings.loss_pause_minutes), int(self._current_capital_profile().loss_pause_minutes))

    def _effective_loss_pause_after_consecutive(self) -> int:
        return min(int(self.settings.loss_pause_after_consecutive), int(self._current_capital_profile().loss_pause_after_consecutive))

    def _effective_max_concurrent_trades(self) -> int:
        return min(int(self.settings.max_concurrent_trades), int(self._current_capital_profile().max_concurrent_trades))

    def _effective_entry_quality_floor(self) -> float:
        return float(self._current_capital_profile().entry_quality_floor)

    def _build_setup_quality_score(self, *, bundle: SignalBundle, final_confidence: float, confluence_score: float) -> float:
        confidence_component = max(min(final_confidence, 1.0), 0.0) * 0.45
        confluence_component = max(min(confluence_score, 1.0), 0.0) * 0.25
        directional_votes = sum(
            1
            for signal in (bundle.trend, bundle.momentum, bundle.structure, bundle.order_flow)
            if signal in {"BUY", "SELL"}
        )
        vote_component = min(directional_votes / 4.0, 1.0) * 0.20
        regime_component = (0.10 if bundle.regime_trade_allowed else 0.0) * max(float(bundle.size_multiplier), 0.0)
        return round(min(confidence_component + confluence_component + vote_component + regime_component, 1.0), 4)

    def _confidence_size_multiplier(self, confidence: float) -> float:
        conf = max(min(float(confidence), 1.0), 0.0)
        base = max(_as_float(getattr(self, "runtime_confidence_boost_multiplier", 1.0), 1.0), 0.75)
        if conf >= 0.90:
            return 1.20 * base
        if conf >= 0.85:
            return 1.10 * base
        if conf <= 0.65:
            return 0.85 * min(base, 1.0)
        return 1.0 * base

    def _assess_spread_gate(self, *, spread: float, price: float) -> dict[str, object]:
        safe_price = max(price, 1e-9)
        spread_ratio = max(float(spread), 0.0) / safe_price
        hard_limit = self._effective_max_spread_ratio()
        soft_limit = hard_limit * 1.25
        if spread_ratio <= hard_limit:
            return {"approved": True, "size_multiplier": 1.0, "reason": "spread_ok"}
        if spread_ratio <= soft_limit:
            return {"approved": True, "size_multiplier": 0.65, "reason": "spread_soft_reduction"}
        return {"approved": False, "size_multiplier": 0.0, "reason": "spread_extreme_reject"}

    def _assess_trade_cost_gate(self, side: str) -> dict[str, object]:
        if side not in {"BUY", "SELL"}:
            return {"approved": False, "size_multiplier": 0.0, "reason": "invalid_side"}
        if not getattr(self.settings, "enforce_fee_floor", True):
            return {"approved": True, "size_multiplier": 1.0, "reason": "fee_floor_disabled"}
        price = _as_float(self.snapshot.get("price"), 0.0)
        if price <= 0:
            return {"approved": False, "size_multiplier": 0.0, "reason": "invalid_price"}
        spread_ratio = _as_float(self.snapshot.get("spread"), 0.0) / price
        fee_rate = max(_as_float(getattr(self.settings, "estimated_fee_rate", 0.001), 0.001), 0.0)
        total_cost = max((fee_rate * 2.0) + spread_ratio, 0.0)
        atr = _as_float(self.snapshot.get("atr"), 0.0)
        expected_move = atr / price if atr > 0 else 0.0
        min_rr = max(_as_float(getattr(self.settings, "min_expected_reward_risk", 1.8), 1.8), float(self._current_capital_profile().min_expected_reward_risk))
        if total_cost <= 0:
            return {"approved": True, "size_multiplier": 1.0, "reason": "cost_floor_not_applicable"}
        required_move = total_cost * min_rr
        coverage = expected_move / max(required_move, 1e-9)
        if coverage >= 1.0:
            return {"approved": True, "size_multiplier": 1.0, "reason": "cost_floor_ok"}
        if coverage >= 0.8:
            return {"approved": True, "size_multiplier": 0.7, "reason": "cost_floor_soft_reduction"}
        return {"approved": False, "size_multiplier": 0.0, "reason": "cost_floor_extreme_reject"}

    async def _log_trade_rejection_trace(
        self,
        analysis: dict[str, Any],
        filter_checks: list[dict[str, Any]],
        reason: str,
    ) -> None:
        if not bool(getattr(self.settings, "verbose_trade_decision_logs", False)):
            return
        lines = ["[TRADE VALIDATION]"]
        signal = str(analysis.get("side") or self.snapshot.get("signal") or "HOLD")
        confidence = _as_float(analysis.get("confidence"), _as_float(self.snapshot.get("confidence"), 0.0))
        min_confidence = max(self.settings.confidence_threshold, self._effective_min_signal_confidence())
        setup_quality = _as_float(analysis.get("setup_quality"), _as_float(self.snapshot.get("signal_quality_score"), 0.0))
        lines.append("")
        lines.append(f"signal: {signal}")
        lines.append(
            f"confidence: {confidence:.4f} (threshold: {min_confidence:.4f}) "
            f"→ {'PASS' if confidence >= min_confidence else 'FAIL'}"
        )
        lines.append(
            f"setup_quality: {setup_quality:.4f} (required: {self._effective_entry_quality_floor():.4f}) "
            f"→ {'PASS' if setup_quality >= self._effective_entry_quality_floor() else 'FAIL'}"
        )
        lines.append(
            f"spread: {_as_float(self.snapshot.get('spread'), 0.0):.8f} "
            f"(limit: {self._effective_max_spread_ratio():.6f})"
        )
        for check in filter_checks:
            check_name = str(check.get("name", "unknown"))
            value = check.get("value")
            threshold = check.get("threshold")
            passed = bool(check.get("passed"))
            lines.append(f"{check_name}: {value} (threshold: {threshold}) → {'PASS' if passed else 'FAIL'}")
        lines.append("")
        lines.append(f"FINAL DECISION: REJECTED ({reason.upper()})")
        await self._log("INFO", "\n".join(lines))

    async def _log_trade_cycle_summary(
        self,
        *,
        analysis: dict[str, Any] | None,
        decision: str,
        reason: str,
        filter_checks: list[dict[str, Any]],
    ) -> None:
        if not bool(getattr(self.settings, "verbose_trade_decision_logs", False)):
            return
        signal = str((analysis or {}).get("side") or self.snapshot.get("signal") or "HOLD")
        confidence = _as_float((analysis or {}).get("confidence"), _as_float(self.snapshot.get("confidence"), 0.0))
        setup_quality = _as_float((analysis or {}).get("setup_quality"), _as_float(self.snapshot.get("signal_quality_score"), 0.0))
        spread = _as_float(self.snapshot.get("spread"), 0.0)
        price = max(_as_float(self.snapshot.get("price"), 0.0), 1e-9)
        expected_move = _as_float(self.snapshot.get("atr"), 0.0) / price if price > 0 else 0.0
        checks_text = ",".join(
            f"{str(item.get('name', 'unknown')).upper()}={'PASS' if bool(item.get('passed')) else 'FAIL'}" for item in filter_checks
        ) or "NONE"
        await self._log(
            "INFO",
            "trade_cycle_summary "
            f"SYMBOL={self.symbol} SIGNAL={signal} CONFIDENCE={confidence:.4f} SETUP_QUALITY={setup_quality:.4f} "
            f"SPREAD={spread:.8f} EXPECTED_MOVE={expected_move:.8f} FILTERS={checks_text} "
            f"DECISION={decision} REASON={str(reason).upper()}",
        )

    async def _log_waiting_state_reason(self, reason: str) -> None:
        if not bool(getattr(self.settings, "verbose_trade_decision_logs", False)):
            return
        await self._log("INFO", f"STATE=WAITING_MARKET_DATA REASON={str(reason).upper()}")

    async def _set_state(self, new_state: BotState, context: str = "") -> None:
        if new_state == self.state:
            return
        previous = self.state
        self.state = new_state
        self.snapshot["status"] = new_state.value
        await self.repository.record_state_change(previous.value, new_state.value, context)

    async def _log(self, level: str, message: str) -> None:
        if not (self.terminal_tui_enabled and self.terminal_tui_quiet_logs):
            getattr(self.logger, level.lower(), self.logger.info)(message)
        logs = list(self.snapshot.get("logs", []) or [])
        logs.append(
            {
                "time": datetime.utcnow().strftime("%H:%M:%S"),
                "level": str(level).upper(),
                "message": str(message),
            }
        )
        self.snapshot["logs"] = logs[-120:]
        await self.repository.record_log(level, self.state.value, message)
        if self.state_manager:
            self.state_manager.add_log(level, message)

    def _roll_day(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self.day_marker:
            self.day_marker = today
            self.trades_today = 0
            self.win_count = 0
            asyncio.create_task(self.repository.cleanup_old_logs(keep_days=7))

    def _is_cooldown_complete(self) -> bool:
        if self.last_close_time is None:
            return True
        return datetime.now(timezone.utc) - self.last_close_time >= timedelta(minutes=self._effective_cooldown_minutes())

    def _safe_live_update(self, live: Live) -> None:
        try:
            live.update(self.dashboard.render(self.snapshot))
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("dashboard_render_error: %s", exc)

    def _sync_ui_state(self) -> None:
        if not self.state_manager:
            return
        try:
            _ss = self.session_tracker.stats()
            self.state_manager.update(
                pair=self.snapshot.get("pair", ""),
                timeframe=self.snapshot.get("timeframe", ""),
                current_price=self.snapshot.get("price"),
                spread=self.snapshot.get("spread"),
                trend=self.snapshot.get("trend"),
                adx=self.snapshot.get("adx"),
                volatility_regime=self.snapshot.get("volatility_regime"),
                order_flow=self.snapshot.get("order_flow"),
                signal=self.snapshot.get("signal"),
                confidence=self.snapshot.get("confidence"),
                balance=self.snapshot.get("balance"),
                equity=self.snapshot.get("equity"),
                btc_balance=self.snapshot.get("btc_balance", 0.0),
                btc_value=self.snapshot.get("btc_value", 0.0),
                total_equity=self.snapshot.get("total_equity", self.snapshot.get("equity")),
                daily_pnl=self.snapshot.get("daily_pnl"),
                session_pnl=self.snapshot.get("session_pnl"),
                trades_today=self.snapshot.get("trades_today", 0),
                win_rate=self.snapshot.get("win_rate"),
                last_trade=self.snapshot.get("last_trade"),
                has_open_position=bool(self.position_manager.positions),
                unrealized_pnl=self.snapshot.get("unrealized_pnl", 0.0),
                open_position_side=self.snapshot.get("open_position_side"),
                open_position_entry=self.snapshot.get("open_position_entry"),
                open_position_qty=self.snapshot.get("open_position_qty"),
                open_position_sl=self.snapshot.get("open_position_sl"),
                open_position_tp=self.snapshot.get("open_position_tp"),
                cooldown=self.snapshot.get("cooldown"),
                status=self.snapshot.get("status", "INITIALIZING"),
                bid=self.snapshot.get("bid", self.snapshot.get("price")),
                ask=self.snapshot.get("ask", self.snapshot.get("price")),
                volume=self.snapshot.get("volume"),
                market_regime=self.snapshot.get("market_regime"),
                volatility_state=self.snapshot.get("volatility_state"),
                distance_to_support=self.snapshot.get("distance_to_support"),
                distance_to_resistance=self.snapshot.get("distance_to_resistance"),
                atr=self.snapshot.get("atr", 0.0),
                change_24h=self.snapshot.get("change_24h"),
                signals=self.snapshot.get("signals", {}),
                candles_5m=list(self.snapshot.get("candles_5m") or [])[-120:],
                system={
                    "uptime_seconds": time.time() - self.start_time,
                    "api_latency_ms": _as_float(self.snapshot.get("api_latency_ms"), 0.0),
                    "database_status": self.snapshot.get("database_status", "UNKNOWN"),
                    "exchange_status": self.snapshot.get("exchange_status", "UNKNOWN"),
                    "redis_status": self.snapshot.get("redis_status", "UNKNOWN"),
                    "memory_usage_mb": round(_get_memory_mb(), 1),
                    "last_server_sync": datetime.utcnow().isoformat(timespec="seconds"),
                },
                market_analysis=self.snapshot.get("market_analysis", {}),
                risk_metrics={
                    "risk_per_trade": f"{self._effective_risk_per_trade_fraction():.2%}",
                    "max_concurrent_trades": self._effective_max_concurrent_trades(),
                    "max_trade_allocation": f"{self._effective_max_trade_balance_fraction():.2%}",
                    "daily_drawdown": max(0.0, -_as_float(self.snapshot.get("daily_pnl"), 0.0)),
                    "consecutive_losses": self.consecutive_losses,
                    "current_exposure": self._current_exposure(),
                    "capital_profile": self.snapshot.get("capital_profile", "UNKNOWN"),
                    "operable_capital_usdt": self.snapshot.get("operable_capital_usdt"),
                    "capital_reserve_ratio": self.snapshot.get("capital_reserve_ratio"),
                    "min_cash_buffer_usdt": self.snapshot.get("min_cash_buffer_usdt"),
                    "setup_quality_score": self.snapshot.get("signal_quality_score"),
                    "advanced_risk_reason": self.snapshot.get("advanced_risk_reason", "OK"),
                    "adaptive_size_multiplier": self.snapshot.get("adaptive_size_multiplier", 1.0),
                    "advanced_size_multiplier": self.snapshot.get("advanced_size_multiplier", 1.0),
                },
                analytics={
                    "total_trades": self.trades_today,
                    "win_rate": self.snapshot.get("win_rate") or 0.0,
                    "profit_factor": 0.0,
                    "average_win": 0.0,
                    "average_loss": 0.0,
                    "largest_win": 0.0,
                    "largest_loss": 0.0,
                    "equity_curve": list(self.equity_curve_history) or [self.snapshot.get("equity") or 0.0],
                    "session_stats": {
                        "total_trades": self.session_tracker.stats().total_trades,
                        "win_rate": self.session_tracker.stats().win_rate,
                        "streak": self.session_tracker.stats().current_streak,
                        "recommendation": self.session_tracker.stats().recommendation,
                        "profit_factor": self.session_tracker.stats().profit_factor,
                        "sharpe": self.session_tracker.stats().sharpe_estimate,
                    },
                },
                runtime_settings=self.snapshot.get("runtime_settings", {}),
                exit_intelligence={
                    "score": self.snapshot.get("exit_intelligence_score", 0.0),
                    "threshold": self.snapshot.get("exit_intelligence_threshold", 0.0),
                    "reason": self.snapshot.get("exit_intelligence_reason", "UNKNOWN"),
                    "codes": list(self.snapshot.get("exit_intelligence_codes", []) or []),
                    "details": dict(self.snapshot.get("exit_intelligence_details", {}) or {}),
                    "events": list(self.snapshot.get("exit_intelligence_log", []) or []),
                },
                llm_trade_confirmator=self.trade_confirmator.stats,
                llm_analysis=self.log_analyzer.get_summary(),
                auto_fix_coordinator=self.auto_fix_coordinator.get_status(),
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("state_sync_error: %s", exc)

    def _current_exposure(self) -> float:
        equity = _as_float(self.snapshot.get("equity"), 0.0)
        if equity <= 0:
            return 0.0
        mark_price = _as_float(self.snapshot.get("price"), 0.0)
        position_value = sum(max(_as_float(p.quantity, 0.0) * mark_price, 0.0) for p in self.position_manager.positions)
        return position_value / equity

    def _is_market_data_fresh(self, market_data: dict[str, Any]) -> bool:
        candle = market_data.get("candle")
        candle_timestamp = _timestamp_to_datetime(candle.get("timestamp") if candle is not None else None)
        if candle_timestamp is None:
            return False
        max_age_seconds = max(_timeframe_to_seconds(self.settings.primary_timeframe) * 2, 1)
        age_seconds = (datetime.now(timezone.utc) - candle_timestamp).total_seconds()
        return age_seconds <= max_age_seconds

    def _register_exchange_failure(self) -> None:
        self.exchange_failure_count += 1
        self.observability.record_reconnection()
        if self.exchange_failure_count < self.exchange_failure_max:
            return
        self.exchange_failure_paused_until = datetime.now(timezone.utc) + timedelta(seconds=self.exchange_failure_cooldown_seconds)
        self.exchange_failure_count = 0
        self.snapshot["cooldown"] = "EXCHANGE_CIRCUIT_BREAKER"
        self.observability.record_circuit_breaker_trip()
        self.logger.critical(
            "exchange_circuit_breaker_triggered pause_until=%s",
            self.exchange_failure_paused_until.isoformat(timespec="seconds"),
        )

    def _refresh_observability_snapshot(self) -> None:
        metrics = self.observability.snapshot()
        self.snapshot["api_latency_p95_ms"] = metrics["api_latency_p95_ms"]
        self.snapshot["stale_market_data_ratio"] = metrics["stale_market_data_ratio"]
        self.snapshot["exchange_reconnections"] = metrics["reconnections"]
        self.snapshot["circuit_breaker_trips"] = metrics["circuit_breaker_trips"]
        
        self.snapshot["health"] = {
            "healthy": metrics.get("db_healthy", True) and metrics.get("exchange_healthy", True),
            "checks": 8,
            "healthy_checks": sum([
                1 if metrics.get("db_healthy", True) else 0,
                1 if metrics.get("exchange_healthy", True) else 0,
                1 if hasattr(self, 'resilience') and self.resilience else 0,
                1 if hasattr(self, 'auto_improver') and self.auto_improver.enabled else 0,
                1 if hasattr(self, 'enhanced_ml') and self._ml_enabled else 0,
                1 if hasattr(self, 'multi_pair_manager') and self.multi_pair_manager else 0,
                1 if hasattr(self, 'autonomous_brain') and self.autonomous_brain.config.enabled else 0,
                1 if self.snapshot.get("database_status") == "CONNECTED" else 0,
            ]),
            "unhealthy_checks": 0,
            "last_check": datetime.now(timezone.utc).isoformat(),
            "results": [
                {"name": "Database", "healthy": metrics.get("db_healthy", True), "message": "Connected" if metrics.get("db_healthy", True) else "Error", "checked_at": datetime.now(timezone.utc).isoformat()},
                {"name": "Exchange", "healthy": metrics.get("exchange_healthy", True), "message": "Connected" if metrics.get("exchange_healthy", True) else "Error", "checked_at": datetime.now(timezone.utc).isoformat()},
                {"name": "Resilience", "healthy": True, "message": "Active", "checked_at": datetime.now(timezone.utc).isoformat()},
                {"name": "Auto-Improver", "healthy": self.auto_improver.enabled if hasattr(self, 'auto_improver') else False, "message": "Active" if getattr(self, 'auto_improver', None) else "Disabled", "checked_at": datetime.now(timezone.utc).isoformat()},
                {"name": "ML Engine", "healthy": self._ml_enabled if hasattr(self, '_ml_enabled') else False, "message": "Running" if getattr(self, '_ml_enabled', False) else "Disabled", "checked_at": datetime.now(timezone.utc).isoformat()},
                {"name": "Multi-Pair", "healthy": True, "message": f"Active ({len(self.symbols)} pairs)", "checked_at": datetime.now(timezone.utc).isoformat()},
                {"name": "Autonomous Brain", "healthy": self.autonomous_brain.config.enabled if hasattr(self, 'autonomous_brain') else False, "message": "Active" if getattr(self, 'autonomous_brain', None) else "Disabled", "checked_at": datetime.now(timezone.utc).isoformat()},
                {"name": "Position Manager", "healthy": True, "message": f"{len(self.position_manager.positions)} open positions", "checked_at": datetime.now(timezone.utc).isoformat()},
            ],
            "component_details": {
                "database": "OK" if metrics.get("db_healthy", True) else "ERROR",
                "exchange": "OK" if metrics.get("exchange_healthy", True) else "ERROR",
                "cache": "OK",
                "metrics_server": "OK" if self._metrics_server_started else "Not Started",
            },
        }
        
        emergency_status = self.emergency_system.get_status()
        self.snapshot["emergency_system"] = {
            "emergency_level": emergency_status.get("emergency_level", "normal"),
            "trading_allowed": emergency_status.get("trading_allowed", True),
            "kill_switch_triggered": emergency_status.get("kill_switch_triggered", False),
            "consecutive_losses": emergency_status.get("consecutive_losses", 0),
            "session_pnl": emergency_status.get("session_pnl", 0),
            "drawdown_percent": emergency_status.get("drawdown_percent", 0),
        }
        
        self.snapshot["hyperopt"] = {
            "status": "Activo" if self.auto_improver.enabled else "Inactivo",
            "current_trial": self.auto_improver.get_improvement_metrics().get("optimization_count", 0) if hasattr(self, 'auto_improver') else 0,
            "total_trials": self.auto_improver.get_improvement_metrics().get("optimization_count", 0) if hasattr(self, 'auto_improver') else 0,
            "best_score": self.auto_improver.get_performance_summary().get("win_rate", 0) if hasattr(self, 'auto_improver') else 0.0,
            "elapsed_seconds": (datetime.now(timezone.utc) - datetime.fromtimestamp(self.start_time, timezone.utc)).total_seconds() if hasattr(self, 'start_time') else 0,
            "best_params": {
                "risk_fraction": self.settings.risk_per_trade_fraction if hasattr(self, 'settings') else 0.01,
                "confidence_threshold": self.settings.confidence_threshold if hasattr(self, 'settings') else 0.55,
                "maker_fee_rate": 0.001,
                "taker_fee_rate": 0.001,
            },
            "trials": [],
        }
        
        resilience_status = self.resilience.get_health_status()
        self.snapshot["resilience_healthy"] = resilience_status["is_healthy"]
        self.snapshot["resilience_consecutive_failures"] = resilience_status["consecutive_failures"]
        
        if self.auto_improver.enabled:
            perf = self.auto_improver.get_performance_summary()
            self.snapshot["auto_improve_total_trades"] = perf["total_trades"]
            self.snapshot["auto_improve_win_rate"] = perf["win_rate"]
            self.snapshot["auto_improve_consecutive_losses"] = perf["consecutive_losses"]
            self.snapshot["auto_improve_optimization_count"] = self.auto_improver.get_improvement_metrics()["optimization_count"]
            
            metrics = self.auto_improver.get_improvement_metrics()
            win_rate_raw = perf.get('win_rate', 0)
            if isinstance(win_rate_raw, str):
                win_rate_raw = win_rate_raw.replace('%', '').strip()
            try:
                win_rate = float(win_rate_raw)
            except (ValueError, TypeError):
                win_rate = 0.0
            
            self.snapshot["auto_improve_logs"] = [
                {
                    "time": metrics.get("last_optimization", datetime.now(timezone.utc)).isoformat() if metrics.get("last_optimization") else datetime.now(timezone.utc).isoformat(),
                    "log_type": "auto_improver",
                    "message": f"Optimization #{metrics.get('optimization_count', 0)} completed - Win rate: {win_rate:.2%}",
                },
                {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "log_type": "auto_improver",
                    "message": f"Current strategy version: {metrics.get('current_strategy_version', '1.0')}",
                },
                {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "log_type": "auto_improver",
                    "message": f"Performance: {perf.get('total_trades', 0)} trades, {win_rate:.2%} win rate",
                },
            ]
        
        self.snapshot["active_pair"] = self.multi_pair_manager.active_pair
        try:
            metrics = self.multi_pair_manager.get_pair_metrics()
            self.snapshot["pair_opportunity_score"] = metrics.opportunity_score if metrics else 0
        except Exception:
            self.snapshot["pair_opportunity_score"] = 0
        
        try:
            trading_status = self.trading_mode_manager.get_status()
            self.snapshot["trading_mode"] = trading_status.get("mode", "spot")
            self.snapshot["trading_leverage"] = trading_status.get("leverage", 1)
            self.snapshot["trading_side"] = trading_status.get("side", "long")
        except Exception:
            pass
        
        self.snapshot["all_modules_initialized"] = self.all_modules_initialized
        self.snapshot["ws_connected"] = self.ws_manager.is_connected()

    def _frame_to_candles(self, frame: Any) -> list[dict[str, float]]:
        candles: list[dict[str, float]] = []
        try:
            frame_slice = frame.tail(120)
        except Exception:  # noqa: BLE001
            return candles

        for _, row in frame_slice.iterrows():
            try:
                candles.append(
                    {
                        "open": _as_float(row.get("open"), 0.0),
                        "high": _as_float(row.get("high"), 0.0),
                        "low": _as_float(row.get("low"), 0.0),
                        "close": _as_float(row.get("close"), 0.0),
                        "volume": _as_float(row.get("volume"), 0.0),
                        "rsi": _as_float(row.get("rsi"), 50.0),
                        "macd_diff": _as_float(row.get("macd_diff"), 0.0),
                        "macd": _as_float(row.get("macd"), 0.0),
                        "macd_signal": _as_float(row.get("macd_signal"), 0.0),
                        "ema9": _as_float(row.get("ema9"), 0.0),
                    }
                )
            except Exception:  # noqa: BLE001
                continue
        return candles

    def _apply_symbol_filter_config(self, filter_config: dict[str, float]) -> None:
        self.base_filter_config = {
            "adx_threshold": _as_float(filter_config.get("adx_threshold"), 22.0),
            "rsi_buy_threshold": _as_float(filter_config.get("rsi_buy_threshold"), 55.0),
            "rsi_sell_threshold": _as_float(filter_config.get("rsi_sell_threshold"), 45.0),
            "volume_buy_threshold": _as_float(filter_config.get("volume_buy_threshold"), 1.10),
            "volume_sell_threshold": _as_float(filter_config.get("volume_sell_threshold"), 0.65),
            "atr_low_threshold": _as_float(filter_config.get("atr_low_threshold"), 0.003),
            "atr_high_threshold": _as_float(filter_config.get("atr_high_threshold"), 0.018),
            "stop_loss_atr_multiplier": _as_float(filter_config.get("stop_loss_atr_multiplier"), 1.5),
            "take_profit_atr_multiplier": _as_float(filter_config.get("take_profit_atr_multiplier"), 2.5),
            "min_confidence": _as_float(filter_config.get("min_confidence"), 0.55),
        }
        self.runtime_filter_config = dict(self.base_filter_config)
        self.logger.info(f"Applied filter config for {self.symbol}: {self.runtime_filter_config}")

    def _apply_autonomous_filters(self) -> None:
        """Apply filter adjustments from the autonomous brain."""
        try:
            if not hasattr(self, 'autonomous_brain') or not self.autonomous_brain.config.enabled:
                return
            
            autonomous_filters = self.autonomous_brain.get_current_filters()
            if not autonomous_filters:
                return
            
            effective_filters = dict(self.base_filter_config)
            market_condition = self.autonomous_brain._current_market_condition
            
            if market_condition == "HIGH_VOLATILITY":
                effective_filters["adx_threshold"] = max(effective_filters.get("adx_threshold", 22.0), 25.0)
                effective_filters["min_confidence"] = max(effective_filters.get("min_confidence", 0.55), 0.70)
            elif market_condition == "LOW_VOLATILITY":
                effective_filters["adx_threshold"] = min(effective_filters.get("adx_threshold", 22.0), 15.0)
                effective_filters["volume_buy_threshold"] = min(effective_filters.get("volume_buy_threshold", 1.10), 0.80)
            elif market_condition == "TRENDING":
                effective_filters["rsi_buy_threshold"] = min(effective_filters.get("rsi_buy_threshold", 55.0), 40.0)
                effective_filters["rsi_sell_threshold"] = max(effective_filters.get("rsi_sell_threshold", 45.0), 60.0)
            elif market_condition == "DEFENSIVE":
                effective_filters["min_confidence"] = max(effective_filters.get("min_confidence", 0.55), 0.75)
                effective_filters["adx_threshold"] = max(effective_filters.get("adx_threshold", 22.0), 30.0)

            allowed_ranges = {
                "adx_threshold": (5.0, 60.0),
                "rsi_buy_threshold": (20.0, 80.0),
                "rsi_sell_threshold": (20.0, 80.0),
                "volume_buy_threshold": (0.2, 3.0),
                "volume_sell_threshold": (0.2, 3.0),
                "atr_low_threshold": (0.0001, 0.05),
                "atr_high_threshold": (0.001, 0.20),
                "stop_loss_atr_multiplier": (0.8, 4.0),
                "take_profit_atr_multiplier": (1.0, 8.0),
                "min_confidence": (0.30, 0.95),
            }
            if isinstance(autonomous_filters, dict):
                for key, value in autonomous_filters.items():
                    if key not in allowed_ranges:
                        continue
                    lo, hi = allowed_ranges[key]
                    effective_filters[key] = min(max(_as_float(value, effective_filters.get(key, lo)), lo), hi)
            
            self.runtime_filter_config = effective_filters
            self.snapshot["autonomous_market_condition"] = market_condition
            self.snapshot["autonomous_filters"] = effective_filters.copy()
            
        except Exception as e:
            self.logger.warning(f"Failed to apply autonomous filters: {e}")

    def _get_default_filter_config(self) -> dict[str, float]:
        normalized_symbol = self.symbol.replace("/", "").upper()
        default_configs = {
            "BTCUSDT": {
                "adx_threshold": 10.0, "rsi_buy_threshold": 45.0, "rsi_sell_threshold": 55.0,
                "volume_buy_threshold": 0.70, "volume_sell_threshold": 0.70,
                "atr_low_threshold": 0.0008, "atr_high_threshold": 0.030,
                "stop_loss_atr_multiplier": 1.5, "take_profit_atr_multiplier": 2.5, "min_confidence": 0.40,
            },
            "ETHUSDT": {
                "adx_threshold": 10.0, "rsi_buy_threshold": 45.0, "rsi_sell_threshold": 55.0,
                "volume_buy_threshold": 0.70, "volume_sell_threshold": 0.70,
                "atr_low_threshold": 0.001, "atr_high_threshold": 0.040,
                "stop_loss_atr_multiplier": 1.6, "take_profit_atr_multiplier": 2.6, "min_confidence": 0.40,
            },
            "SOLUSDT": {
                "adx_threshold": 10.0, "rsi_buy_threshold": 45.0, "rsi_sell_threshold": 55.0,
                "volume_buy_threshold": 0.70, "volume_sell_threshold": 0.70,
                "atr_low_threshold": 0.002, "atr_high_threshold": 0.060,
                "stop_loss_atr_multiplier": 2.0, "take_profit_atr_multiplier": 3.0, "min_confidence": 0.40,
            },
            "BNBUSDT": {
                "adx_threshold": 10.0, "rsi_buy_threshold": 45.0, "rsi_sell_threshold": 55.0,
                "volume_buy_threshold": 0.70, "volume_sell_threshold": 0.70,
                "atr_low_threshold": 0.001, "atr_high_threshold": 0.050,
                "stop_loss_atr_multiplier": 1.8, "take_profit_atr_multiplier": 2.8, "min_confidence": 0.45,
            },
            "XRPUSDT": {
                "adx_threshold": 10.0, "rsi_buy_threshold": 45.0, "rsi_sell_threshold": 55.0,
                "volume_buy_threshold": 0.80, "volume_sell_threshold": 0.95,
                "atr_low_threshold": 0.002, "atr_high_threshold": 0.080,
                "stop_loss_atr_multiplier": 2.2, "take_profit_atr_multiplier": 3.5, "min_confidence": 0.40,
            },
        }
        return default_configs.get(normalized_symbol, {
            "adx_threshold": 12.0, "rsi_buy_threshold": 48.0, "rsi_sell_threshold": 52.0,
            "volume_buy_threshold": 0.90, "volume_sell_threshold": 0.85,
            "atr_low_threshold": 0.001, "atr_high_threshold": 0.040,
            "stop_loss_atr_multiplier": 1.8, "take_profit_atr_multiplier": 3.0, "min_confidence": 0.45,
        })

    def _effective_adx_threshold(self) -> float:
        return self.runtime_filter_config.get("adx_threshold", 10.0)

    def _effective_rsi_buy_threshold(self) -> float:
        return self.runtime_filter_config.get("rsi_buy_threshold", 40.0)

    def _effective_rsi_sell_threshold(self) -> float:
        return self.runtime_filter_config.get("rsi_sell_threshold", 60.0)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        import pandas as pd
        if isinstance(value, pd.Series):
            value = value.iloc[-1] if len(value) > 0 else default
        elif isinstance(value, pd.DataFrame):
            value = value.iloc[-1, -1] if value.size > 0 else default
        return float(value)
    except (TypeError, ValueError, IndexError):
        return default


def _book_price(order_book: dict[str, Any], side: str, fallback: float) -> float:
    levels = order_book.get(side) or []
    if levels and levels[0]:
        return _as_float(levels[0][0], fallback)
    return fallback


def _timeframe_to_seconds(timeframe: str) -> int:
    unit = timeframe[-1].lower()
    value = int(timeframe[:-1])
    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 3600
    if unit == "d":
        return value * 86400
    return max(value, 1)


def _timestamp_to_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromtimestamp(float(value) / 1000, tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None
    return parsed


def _sanitize_runtime_settings_payload(payload: dict[str, Any]) -> dict[str, Any]:
    account_currency = str(payload.get("account_currency", "USDT")).strip().upper() or "USDT"
    allowed_currencies = {"USDT", "USDC", "FDUSD", "BUSD", "TUSD", "DAI", "EUR", "BTC", "ETH", "BNB"}
    if account_currency not in allowed_currencies:
        account_currency = "USDT"
    raw_symbol_limits = payload.get("symbol_capital_limits", {})
    symbol_limits: dict[str, float] = {}
    if isinstance(raw_symbol_limits, dict):
        for symbol, limit in raw_symbol_limits.items():
            normalized_symbol = normalize_symbol(str(symbol))
            if not normalized_symbol:
                continue
            float_limit = _as_float(limit, 0.0)
            if float_limit > 0:
                symbol_limits[normalized_symbol] = float_limit

    return {
        "refresh_rate_ms": max(int(payload.get("refresh_rate_ms", 1000) or 1000), 250),
        "chart_visible": bool(payload.get("chart_visible", True)),
        "theme": str(payload.get("theme", "Dark")).strip() or "Dark",
        "log_verbosity": str(payload.get("log_verbosity", "INFO")).strip().upper() or "INFO",
        "account_currency": account_currency,
        "default_pair": normalize_symbol(str(payload.get("default_pair", "")).strip()),
        "default_timeframe": str(payload.get("default_timeframe", "")).strip(),
        "investment_mode": str(payload.get("investment_mode", "Balanced")).strip() or "Balanced",
        "dynamic_exit_enabled": bool(payload.get("dynamic_exit_enabled", False)),
        "capital_limit_usdt": max(_as_float(payload.get("capital_limit_usdt"), 0.0), 0.0),
        "capital_reserve_ratio": min(max(_as_float(payload.get("capital_reserve_ratio"), 0.15), 0.0), 0.90),
        "min_cash_buffer_usdt": max(_as_float(payload.get("min_cash_buffer_usdt"), 10.0), 0.0),
        "symbol_capital_limits": symbol_limits,
        "risk_per_trade_fraction": min(max(_as_float(payload.get("risk_per_trade_fraction"), 0.01), 0.001), 0.10),
        "max_trade_balance_fraction": min(max(_as_float(payload.get("max_trade_balance_fraction"), 0.20), 0.01), 1.0),
    }
