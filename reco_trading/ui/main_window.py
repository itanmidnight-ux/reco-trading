from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QTimer
from PySide6.QtWidgets import QGraphicsOpacityEffect, QMainWindow, QMessageBox, QTabWidget, QWidget, QVBoxLayout

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
        self.setMinimumSize(1100, 720)
        self.state_manager = state_manager

        try:
            from reco_trading.ui.theme import app_stylesheet

            self.setStyleSheet(app_stylesheet())
        except Exception as exc:  # noqa: BLE001
            print(f"Theme loading failed: {exc}")

        self.tabs = QTabWidget()
        self.tabs.setMovable(True)
        self.tabs.setDocumentMode(True)

        self.dashboard_tab = DashboardTab(state_manager=state_manager)
        self.trades_tab = TradesTab()
        self.market_tab = MarketTab()
        self.analytics_tab = AnalyticsTab()
        self.logs_tab = LogsTab()
        self.risk_tab = RiskTab()
        self.settings_tab = SettingsTab()
        self.system_tab = SystemTab()

        self.tabs.addTab(self.dashboard_tab, "Dashboard")
        self.tabs.addTab(self.trades_tab, "Trades")
        self.tabs.addTab(self.market_tab, "Market")
        self.tabs.addTab(self.analytics_tab, "Analytics")
        self.tabs.addTab(self.logs_tab, "Logs")
        self.tabs.addTab(self.risk_tab, "Risk")
        self.tabs.addTab(self.settings_tab, "Settings")
        self.tabs.addTab(self.system_tab, "System")

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.tabs)
        self.setCentralWidget(central)

        self.tabs.currentChanged.connect(self._animate_current_tab)

        state_manager.state_changed.connect(self._on_state)
        state_manager.trade_added.connect(self.trades_tab.add_trade)
        state_manager.log_added.connect(self.logs_tab.add_log)
        state_manager.notification.connect(self._notify)
        self.settings_tab.settings_changed.connect(self._on_ui_settings)

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._refresh_from_snapshot)
        self.refresh_timer.start(1000)

    def _on_state(self, state: dict) -> None:
        for tab in (
            self.dashboard_tab,
            self.trades_tab,
            self.market_tab,
            self.analytics_tab,
            self.logs_tab,
            self.risk_tab,
            self.system_tab,
        ):
            try:
                tab.update_state(state or {})
            except Exception:
                continue

    def _refresh_from_snapshot(self) -> None:
        self._on_state(self.state_manager.snapshot() or {})

    def _on_ui_settings(self, settings: dict) -> None:
        self.refresh_timer.setInterval(int(settings.get("refresh_rate_ms", 1000)))
        self.dashboard_tab.chart_panel.setVisible(bool(settings.get("chart_visible", True)))

    def _notify(self, title: str, message: str) -> None:
        QMessageBox.information(self, title, message)

    def _animate_current_tab(self, index: int) -> None:
        widget = self.tabs.widget(index)
        if not widget:
            return
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
        animation = QPropertyAnimation(effect, b"opacity", widget)
        animation.setDuration(220)
        animation.setStartValue(0.55)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.start()
        self._current_animation = animation
