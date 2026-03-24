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
from reco_trading.strategy.signal_engine import SignalBundle, SignalEngine
from reco_trading.ui.dashboard import TerminalDashboard

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
        except Exception:
            pass
        return 0.0


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
        self.repository = Repository(settings.postgres_dsn)
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
        self.confluence = TimeframeConfluence()

        self.state_manager = state_manager
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

        self._cached_frame5: Any | None = None
        self._cached_frame15: Any | None = None
        self._last_primary_indicator_ts: datetime | None = None
        self._last_confirmation_indicator_ts: datetime | None = None

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
        }

    async def run(self) -> None:
        try:
            await self._set_state(BotState.INITIALIZING, "initialize_settings")
            await self.repository.setup()
            self.snapshot["database_status"] = "CONNECTED"
            self.observability.update_health(db_healthy=True)
            if self.settings.observability_enabled and not self._metrics_server_started:
                start_metrics_server(self.observability, self.settings.observability_bind_host, self.settings.observability_port)
                self._metrics_server_started = True
            runtime_bundle = await self.repository.get_runtime_settings()
            runtime_settings = runtime_bundle.get("ui_runtime_settings", {}) if isinstance(runtime_bundle, dict) else {}
            if isinstance(runtime_settings, dict) and runtime_settings:
                await self._apply_runtime_settings(runtime_settings, persist=False)
            await self._set_state(BotState.CONNECTING_EXCHANGE, "connect_exchange")
            await self.client.sync_time()
            await self._set_state(BotState.SYNCING_SYMBOL, "sync_symbol")
            await self._set_state(BotState.SYNCING_RULES, "sync_exchange_rules")
            await self.order_manager.sync_rules()
            self.snapshot["exchange_status"] = "CONNECTED"
            self.observability.update_health(exchange_healthy=True)
            await self._reconcile_open_positions()
            await self._set_state(BotState.WAITING_MARKET_DATA, "ready")
            self._sync_ui_state()

            with Live(self.dashboard.render(self.snapshot), refresh_per_second=2, transient=False) as live:
                while True:
                    try:
                        await self._process_control_requests()
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
                            await self._set_state(BotState.WAITING_MARKET_DATA, "stale_market_data")
                            self.snapshot["cooldown"] = "STALE_MARKET_DATA"
                            self._sync_ui_state()
                            self._safe_live_update(live)
                            await self._sleep_with_responsiveness(self.settings.loop_sleep_seconds)
                            continue
                        await self._set_state(BotState.ANALYZING_MARKET, "analyze_market")
                        analysis = await self.analyze_market(market_data)
                        self._update_snapshot(market_data, analysis)

                        if await self.validate_trade_conditions(analysis):
                            intelligence = self.market_intelligence.evaluate(str(analysis.get("side", "HOLD")), market_data)
                            self._apply_market_intelligence_snapshot(intelligence)
                            if intelligence.get("approved"):
                                await self.execute_trade(analysis, market_data, float(intelligence.get("size_multiplier", 1.0)))
                            else:
                                await self._set_state(BotState.WAITING_MARKET_DATA, str(intelligence.get("reason", "market_intelligence")))

                        await self.manage_open_position(market_data)
                        self.observability.record_loop(stale_market_data=False)
                        self.exchange_failure_count = 0
                        self.observability.update_health(exchange_healthy=True)
                        self._refresh_observability_snapshot()
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
                        await self._log("ERROR", f"exchange_error={exc}")
                        await self.repository.record_error(self.state.value, "exchange", str(exc))
                        self.snapshot["status"] = BotState.ERROR.value
                        self._sync_ui_state()
                        self._safe_live_update(live)
                        await self._sleep_with_responsiveness(self.settings.loop_sleep_seconds)
                    except Exception as exc:  # noqa: BLE001
                        await self._set_state(BotState.ERROR, "runtime_error")
                        self.observability.record_error("runtime")
                        await self._log("ERROR", f"runtime_error={exc}")
                        await self.repository.record_error(self.state.value, "runtime", str(exc))
                        self.snapshot["status"] = BotState.ERROR.value
                        self._safe_live_update(live)
                        await self._sleep_with_responsiveness(self.settings.loop_sleep_seconds)
        finally:
            await self.client.close()
            await self.repository.close()

    async def fetch_market_data(self) -> dict[str, Any]:
        raw_frame5, raw_frame15 = await asyncio.gather(
            self.market_stream.fetch_frame(self.settings.primary_timeframe),
            self.market_stream.fetch_frame(self.settings.confirmation_timeframe),
        )

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
        bundle: SignalBundle = self.signal_engine.generate(market_data["frame5"], market_data["frame15"])
        explained = self.confidence_model.explain(bundle)
        side = str(explained["side"])
        confidence = float(explained["confidence"])
        grade = str(explained["grade"])
        raw_side = side
        conf_result = self.confluence.evaluate(market_data["frame5"], market_data["frame15"])
        if conf_result.aligned:
            final_confidence = min(confidence * 1.08, 0.99)
        else:
            penalty = max(conf_result.score, 0.50)
            final_confidence = confidence * penalty
        if getattr(self.settings, "spot_only_mode", True) and side == "SELL" and not self._can_execute_spot_sell():
            side = "HOLD"

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
        confidence = float(analysis["confidence"])
        side = str(analysis.get("side", "HOLD")).upper()
        profile = self._current_capital_profile()
        min_confidence = max(self.settings.confidence_threshold, self._effective_min_signal_confidence())
        if side == "HOLD":
            await self._set_state(BotState.WAITING_MARKET_DATA, "hold_signal")
            self.snapshot["cooldown"] = "HOLD_SIGNAL"
            self._set_decision_gating("hold_signal", "HOLD")
            return False

        if getattr(self.settings, "spot_only_mode", True) and side == "SELL" and not self._can_execute_spot_sell():
            await self._set_state(BotState.WAITING_MARKET_DATA, "spot_short_blocked")
            self.snapshot["cooldown"] = "SPOT_SHORT_BLOCKED"
            self._set_decision_gating("spot_short_blocked", "HOLD")
            return False

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

        if self.trading_paused_by_drawdown:
            await self._set_state(BotState.PAUSED, "max_drawdown")
            self.snapshot["cooldown"] = "MAX_DRAWDOWN"
            self._set_decision_gating("max_drawdown", "HOLD")
            return False

        if self.pause_trading_until and datetime.now(timezone.utc) < self.pause_trading_until:
            await self._set_state(BotState.PAUSED, "loss_protection_pause")
            self.snapshot["cooldown"] = f"PAUSED until {self.pause_trading_until.isoformat(timespec='seconds')}"
            self._set_decision_gating("loss_protection_pause", "HOLD")
            return False

        if not self._is_cooldown_complete():
            await self._set_state(BotState.COOLDOWN, "cooldown_active")
            self.snapshot["cooldown"] = "ACTIVE"
            self._set_decision_gating("cooldown_active", "HOLD")
            return False

        self.risk_manager.max_trades_per_day = self._effective_max_trades_per_day()
        risk = self.risk_manager.validate(
            balance=usdt_balance,
            daily_pnl=session_pnl,
            trades_today=self.trades_today,
            confidence=confidence,
            confidence_threshold=min_confidence,
        )
        if not risk.approved:
            await self._set_state(BotState.PAUSED if risk.reason == "RISK_PAUSE" else BotState.WAITING_MARKET_DATA, risk.reason)
            self.snapshot["cooldown"] = risk.reason
            self._set_decision_gating(risk.reason, "HOLD")
            return False

        volatility_ratio = _as_float(self.snapshot.get("atr"), 0.0) / max(current_price, 1e-9)
        advanced = self.advanced_risk_manager.evaluate(
            daily_pnl=session_pnl,
            starting_equity=max(_as_float(self.starting_equity, total_equity), 1.0),
            consecutive_losses=self.consecutive_losses,
            current_equity=total_equity,
            peak_equity=max(_as_float(self.equity_peak, total_equity), 1.0),
            volatility_ratio=volatility_ratio,
        )
        if not advanced.approved:
            await self._set_state(BotState.PAUSED, advanced.reason)
            self.snapshot["cooldown"] = advanced.reason
            if advanced.pause_trading:
                self.pause_trading_until = datetime.now(timezone.utc) + timedelta(minutes=self._effective_loss_pause_minutes())
            self._set_decision_gating(advanced.reason, "HOLD")
            return False

        self.snapshot["advanced_risk_reason"] = advanced.reason
        self.snapshot["advanced_size_multiplier"] = advanced.size_multiplier

        session_stats = self.session_tracker.stats()
        if session_stats.recommendation == "PAUSE":
            await self._set_state(BotState.PAUSED, "session_tracker_pause")
            self.snapshot["cooldown"] = "SESSION_TRACKER_PAUSE"
            self._set_decision_gating("session_tracker_pause", "HOLD")
            return False
        self.snapshot["session_streak"] = session_stats.current_streak
        self.snapshot["session_recommendation"] = session_stats.recommendation

        setup_quality = _as_float(analysis.get("setup_quality"), _as_float(self.snapshot.get("signal_quality_score"), 0.0))
        if setup_quality < self._effective_entry_quality_floor():
            await self._set_state(BotState.WAITING_MARKET_DATA, "setup_quality_too_low")
            self.snapshot["cooldown"] = "SETUP_QUALITY_TOO_LOW"
            self._set_decision_gating("setup_quality_too_low", "HOLD")
            return False

        if not self._trade_costs_are_acceptable(side):
            await self._set_state(BotState.WAITING_MARKET_DATA, "trade_costs_too_high")
            self.snapshot["cooldown"] = "TRADE_COSTS_TOO_HIGH"
            self._set_decision_gating("trade_costs_too_high", "HOLD")
            return False

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
            if not portfolio_check.approved:
                await self._set_state(BotState.PAUSED, portfolio_check.reason)
                self.snapshot["cooldown"] = portfolio_check.reason.upper()
                self._set_decision_gating(portfolio_check.reason, "HOLD")
                return False

        self.snapshot["capital_profile"] = profile.name

        self.snapshot["cooldown"] = "READY"
        self._set_decision_gating("ready", side)
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
        spread_ratio = spread / max(price, 1e-9)
        if spread_ratio > self._effective_max_spread_ratio():
            await self._set_state(BotState.WAITING_MARKET_DATA, "spread_too_wide")
            return

        if not self._pullback_confirmed(bundle, side, market_data):
            await self._set_state(BotState.WAITING_MARKET_DATA, "pullback_unconfirmed")
            return

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
        qty = self.calculate_position_size(
            price,
            stop_loss,
            atr,
            float(bundle.size_multiplier)
            * max(float(intelligence_size_multiplier), 0.1)
            * max(_as_float(self.snapshot.get("advanced_size_multiplier"), 1.0), 0.1)
            * self._confidence_size_multiplier(_as_float(analysis.get("confidence"), 0.0)),
            risk_fraction_override=trade_controls["risk_per_trade_fraction"],
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
        except Exception:
            return None

        last_close = _as_float(closes.iloc[-1], current_price)
        prev_close = _as_float(closes.iloc[-2], last_close)
        prev2_close = _as_float(closes.iloc[-3], prev_close)
        recent_swing_high = max(float(highs.iloc[-7:-1].max()), current_price)
        recent_swing_low = min(float(lows.iloc[-7:-1].min()), current_price)
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

        if pnl > 0:
            self.win_count += 1
        self.snapshot["win_rate"] = self.win_count / self.trades_today if self.trades_today else None
        self.snapshot["last_trade"] = f"{position.side} {exit_reason} pnl={pnl:.4f}"

        await self._log("INFO", f"position_closed trade_id={position.trade_id} reason={exit_reason} exit_price={exit_price:.8f} pnl={pnl:.8f}")
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

    def _pullback_confirmed(self, bundle: SignalBundle, side: str, market_data: dict[str, Any]) -> bool:
        """
        Confirmación de entrada con momentum mínimo.
        No bloquea en tendencia. Solo rechaza si la vela más reciente va
        fuertemente en contra de la señal (2 velas consecutivas contrarias).
        """
        frame5 = market_data.get("frame5")
        if frame5 is None or len(frame5) < 4:
            return True

        row = frame5.iloc[-1]
        prev1 = frame5.iloc[-2]
        prev2 = frame5.iloc[-3]

        close = _as_float(row.get("close"), 0.0)
        open_ = _as_float(row.get("open"), close)
        p1c = _as_float(prev1.get("close"), close)
        p2c = _as_float(prev2.get("close"), close)
        atr = _as_float(row.get("atr"), close * 0.005)

        if side == "BUY":
            two_consecutive_bear = (p1c < _as_float(prev1.get("open"), p1c)) and (p2c < _as_float(prev2.get("open"), p2c))
            strong_move_down = (p1c - close) > (0.5 * atr)
            return not (two_consecutive_bear and strong_move_down)

        two_consecutive_bull = (p1c > _as_float(prev1.get("open"), p1c)) and (p2c > _as_float(prev2.get("open"), p2c))
        strong_move_up = (close - p1c) > (0.5 * atr)
        return not (two_consecutive_bull and strong_move_up)

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

    async def _fetch_balances(self) -> tuple[float, float, str, str]:
        payload = await self.client.fetch_balance()
        base_asset, quote_asset = split_symbol(self.symbol)
        quote_asset = quote_asset or "USDT"

        total_balances = payload.get("total") if isinstance(payload, dict) else None
        if isinstance(total_balances, dict):
            quote_balance = _as_float(total_balances.get(quote_asset), 0.0)
            base_balance = _as_float(total_balances.get(base_asset), 0.0)
            return quote_balance, base_balance, quote_asset, base_asset

        quote = payload.get(quote_asset) if isinstance(payload, dict) else {}
        base = payload.get(base_asset) if isinstance(payload, dict) else {}
        quote_balance = _as_float((quote or {}).get("free"), 0.0)
        base_balance = _as_float((base or {}).get("free"), 0.0)
        return quote_balance, base_balance, quote_asset, base_asset

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

    async def _refresh_account_snapshot(self, current_price: float | None = None) -> None:
        quote_balance, base_balance, quote_asset, _base_asset = await self._fetch_balances()
        reference_price = max(_as_float(current_price, _as_float(self.snapshot.get("price"), 0.0)), 0.0)
        base_value_quote = float(base_balance * reference_price)
        total_equity_quote = float(quote_balance + base_value_quote)
        quote_to_usdt_rate = await self._quote_to_usdt_rate(quote_asset)
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
        quote_balance, base_balance, quote_asset, _base_asset = await self._fetch_balances()
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
        if persist:
            await self.repository.set_runtime_setting("ui_runtime_settings", {**sanitized, **auto_profile})

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

    def _trade_costs_are_acceptable(self, side: str) -> bool:
        if side not in {"BUY", "SELL"}:
            return False
        if not getattr(self.settings, "enforce_fee_floor", True):
            return True
        price = _as_float(self.snapshot.get("price"), 0.0)
        if price <= 0:
            return False
        spread_ratio = _as_float(self.snapshot.get("spread"), 0.0) / price
        fee_rate = max(_as_float(getattr(self.settings, "estimated_fee_rate", 0.001), 0.001), 0.0)
        total_cost = max((fee_rate * 2.0) + spread_ratio, 0.0)
        atr = _as_float(self.snapshot.get("atr"), 0.0)
        expected_move = atr / price if atr > 0 else 0.0
        min_rr = max(_as_float(getattr(self.settings, "min_expected_reward_risk", 1.8), 1.8), float(self._current_capital_profile().min_expected_reward_risk))
        if total_cost <= 0:
            return True
        return expected_move >= (total_cost * min_rr)

    async def _set_state(self, new_state: BotState, context: str = "") -> None:
        if new_state == self.state:
            return
        previous = self.state
        self.state = new_state
        self.snapshot["status"] = new_state.value
        await self.repository.record_state_change(previous.value, new_state.value, context)

    async def _log(self, level: str, message: str) -> None:
        getattr(self.logger, level.lower(), self.logger.info)(message)
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


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
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
        "default_pair": str(payload.get("default_pair", "")).strip(),
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
