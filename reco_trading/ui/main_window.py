from __future__ import annotations

import time

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QTimer
from PySide6.QtWidgets import QGraphicsOpacityEffect, QMainWindow, QMessageBox, QTabWidget

from reco_trading.ui.state_manager import StateManager
from reco_trading.ui.tabs.alerts_tab import AlertsTab
from reco_trading.ui.tabs.analytics_tab import AnalyticsTab
from reco_trading.ui.tabs.cache_tab import CacheTab
from reco_trading.ui.tabs.dashboard_tab import DashboardTab
from reco_trading.ui.tabs.ml_intelligence_tab import MLIntelligenceTab
from reco_trading.ui.tabs.health_tab import HealthTab
from reco_trading.ui.tabs.hyperopt_tab import HyperoptTab
from reco_trading.ui.tabs.intel_log_tab import IntelLogTab
from reco_trading.ui.tabs.logs_tab import LogsTab
from reco_trading.ui.tabs.market_tab import MarketTab
from reco_trading.ui.tabs.risk_tab import RiskTab
from reco_trading.ui.tabs.settings_tab import SettingsTab
from reco_trading.ui.tabs.strategy_tab import StrategyTab
from reco_trading.ui.tabs.system_tab import SystemTab
from reco_trading.ui.tabs.trades_tab import TradesTab


class MainWindow(QMainWindow):
    def __init__(self, state_manager: StateManager) -> None:
        super().__init__()
        self.setWindowTitle("Reco Trading Professional Terminal")
        self.resize(1520, 940)
        self.state_manager = state_manager
        try:
            from reco_trading.ui.theme import app_stylesheet

            self.setStyleSheet(app_stylesheet())
        except Exception as exc:  # noqa: BLE001
            print(f"Theme loading failed: {exc}")

        tabs = QTabWidget()
        self.dashboard_tab = DashboardTab(state_manager=state_manager)
        self.trades_tab = TradesTab()
        self.market_tab = MarketTab()
        self.analytics_tab = AnalyticsTab()
        self.strategy_tab = StrategyTab()
        self.freqai_tab = MLIntelligenceTab(state_manager=state_manager)
        self.hyperopt_tab = HyperoptTab(state_manager=state_manager)
        self.cache_tab = CacheTab(state_manager=state_manager)
        self.health_tab = HealthTab(state_manager=state_manager)
        self.logs_tab = LogsTab(state_manager=state_manager)
        self.intel_log_tab = IntelLogTab()
        self.risk_tab = RiskTab()
        self.settings_tab = SettingsTab()
        self.system_tab = SystemTab()

        tabs.addTab(self.dashboard_tab, "Dashboard")
        tabs.addTab(self.trades_tab, "Trades")
        tabs.addTab(self.market_tab, "Market")
        tabs.addTab(self.analytics_tab, "Analytics")
        tabs.addTab(self.strategy_tab, "Strategy")
        tabs.addTab(self.freqai_tab, "RecoAI")
        tabs.addTab(self.hyperopt_tab, "Optimizer")
        tabs.addTab(self.cache_tab, "Cache")
        tabs.addTab(self.health_tab, "Health")
        tabs.addTab(self.logs_tab, "Logs")
        tabs.addTab(self.intel_log_tab, "Intel Log")
        tabs.addTab(self.risk_tab, "Risk")
        tabs.addTab(self.settings_tab, "Settings")
        tabs.addTab(self.system_tab, "System")
        self.setCentralWidget(tabs)
        self.tabs = tabs
        self.tab_fade = QPropertyAnimation(self, b"windowOpacity", self)
        self.tabs.currentChanged.connect(self._animate_current_tab)

        state_manager.state_changed.connect(self._on_state)
        state_manager.trade_added.connect(self.trades_tab.add_trade)
        state_manager.log_added.connect(self.logs_tab.add_log)
        state_manager.notification.connect(self._notify)
        self.settings_tab.settings_changed.connect(self._on_ui_settings)

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._refresh_from_snapshot)
        self.refresh_timer.start(250)
        self._last_state_event_at = 0.0
        self._last_ui_render_ms = 0.0
        self._ui_lag_detected = False

    def _on_state(self, state: dict) -> None:
        self._last_state_event_at = time.monotonic()
        enriched_state = self._decorate_state(state)
        started_at = time.perf_counter()
        for tab in (
            self.dashboard_tab,
            self.trades_tab,
            self.market_tab,
            self.analytics_tab,
            self.strategy_tab,
            self.freqai_tab,
            self.hyperopt_tab,
            self.cache_tab,
            self.health_tab,
            self.logs_tab,
            self.intel_log_tab,
            self.risk_tab,
            self.settings_tab,
            self.system_tab,
        ):
            try:
                tab.update_state(enriched_state)
            except Exception:
                continue
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        self._last_ui_render_ms = elapsed_ms
        self._ui_lag_detected = elapsed_ms >= max(120.0, self.refresh_timer.interval() * 0.8)

    def _refresh_from_snapshot(self) -> None:
        if (time.monotonic() - self._last_state_event_at) < 0.25:
            return
        self._on_state(self.state_manager.snapshot())

    def _on_ui_settings(self, settings: dict) -> None:
        self.refresh_timer.setInterval(int(settings.get("refresh_rate_ms", 1000)))
        self.dashboard_tab.chart_panel.setVisible(bool(settings.get("chart_visible", True)))
        self.state_manager.push_runtime_settings(settings)

    def _notify(self, title: str, message: str) -> None:
        QMessageBox.information(self, title, message)

    def _animate_current_tab(self, index: int) -> None:
        widget = self.tabs.widget(index)
        if not widget:
            return
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
        animation = QPropertyAnimation(effect, b"opacity", widget)
        animation.setDuration(260)
        animation.setStartValue(0.45)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.start()
        self._current_animation = animation

    def _decorate_state(self, state: dict) -> dict:
        decorated = dict(state)
        system = dict(decorated.get("system", {}) or {})
        staleness_ms = max((time.monotonic() - self._last_state_event_at) * 1000, 0.0) if self._last_state_event_at else 0.0
        system.update(
            {
                "ui_render_ms": round(self._last_ui_render_ms, 2),
                "ui_staleness_ms": round(staleness_ms, 2),
                "ui_lag_detected": self._ui_lag_detected,
                "ui_refresh_interval_ms": self.refresh_timer.interval(),
            }
        )
        decorated["system"] = system
        
        if "cache" not in decorated:
            decorated["cache"] = {
                "enabled": True,
                "size": decorated.get("cache_size", 0),
                "hits": decorated.get("cache_hits", 0),
                "misses": decorated.get("cache_misses", 0),
                "hit_rate": decorated.get("cache_hit_rate", 0),
                "evictions": decorated.get("cache_evictions", 0),
                "ohlcv": {"ttl": 30, "symbols": len(decorated.get("symbols", []))},
                "prefetch": {"enabled": True, "symbols": len(decorated.get("symbols", [])), "interval": 25},
                "providers": [{"name": "Binance", "status": "Active", "last_update": decorated.get("last_market_update", "N/A"), "errors": 0}],
            }
        
        if "health" not in decorated:
            exchange_status = decorated.get("exchange_status", "CONNECTED")
            db_status = decorated.get("database_status", "CONNECTED")
            decorated["health"] = {
                "healthy": exchange_status == "CONNECTED" and db_status == "CONNECTED",
                "checks": 4,
                "healthy_checks": sum(1 for s in [exchange_status, db_status] if s == "CONNECTED"),
                "unhealthy_checks": sum(1 for s in [exchange_status, db_status] if s != "CONNECTED"),
                "last_check": decorated.get("last_update", "-"),
                "results": [
                    {"name": "Exchange", "healthy": exchange_status == "CONNECTED", "message": exchange_status, "checked_at": decorated.get("last_update", "-")},
                    {"name": "Database", "healthy": db_status == "CONNECTED", "message": db_status, "checked_at": decorated.get("last_update", "-")},
                    {"name": "Market Data", "healthy": decorated.get("price", 0) > 0, "message": f"Price: {decorated.get('price', 0)}", "checked_at": decorated.get("last_update", "-")},
                    {"name": "ML Engine", "healthy": decorated.get("ml_direction") is not None, "message": decorated.get("ml_direction", "Initializing"), "checked_at": decorated.get("last_update", "-")},
                ],
                "component_details": {"database": db_status, "exchange": exchange_status, "cache": "Active", "metrics_server": "Active"},
            }
        
        if "ml_intelligence" not in decorated:
            decorated["ml_intelligence"] = {
                "status": "Activo",
                "model_type": "Ensemble (Momentum, Trend, Volume, Pattern, Sentiment)",
                "training_samples": decorated.get("ml_training_samples", 0),
                "last_train": decorated.get("ml_last_train", "En tiempo real"),
                "next_train": "Continuo",
                "metrics": decorated.get("ml_metrics", {}),
                "features": [
                    {"name": "momentum", "type": "technical", "importance": 1.0},
                    {"name": "trend", "type": "technical", "importance": 1.0},
                    {"name": "volume", "type": "volume", "importance": 0.8},
                    {"name": "pattern", "type": "candlestick", "importance": 0.8},
                    {"name": "sentiment", "type": "market", "importance": 1.0},
                ],
            }
        
        return decorated
