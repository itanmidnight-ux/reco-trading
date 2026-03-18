from __future__ import annotations

import time

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QTimer
from PySide6.QtWidgets import QGraphicsOpacityEffect, QMainWindow, QMessageBox, QTabWidget

from reco_trading.ui.state_manager import StateManager
from reco_trading.ui.tabs.analytics_tab import AnalyticsTab
from reco_trading.ui.tabs.dashboard_tab import DashboardTab
from reco_trading.ui.tabs.logs_tab import LogsTab
from reco_trading.ui.tabs.market_tab import MarketTab
from reco_trading.ui.tabs.risk_tab import RiskTab
from reco_trading.ui.tabs.settings_tab import SettingsTab
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
        self.logs_tab = LogsTab(state_manager=state_manager)
        self.risk_tab = RiskTab()
        self.settings_tab = SettingsTab()
        self.system_tab = SystemTab()

        tabs.addTab(self.dashboard_tab, "Dashboard")
        tabs.addTab(self.trades_tab, "Trades")
        tabs.addTab(self.market_tab, "Market")
        tabs.addTab(self.analytics_tab, "Analytics")
        tabs.addTab(self.logs_tab, "Logs")
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
            self.logs_tab,
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
        return decorated
