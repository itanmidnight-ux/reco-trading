from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QMessageBox, QTabWidget

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
        self.setWindowTitle("Reco Trading Dashboard")
        self.resize(1400, 900)
        self.state_manager = state_manager

        tabs = QTabWidget()
        self.dashboard_tab = DashboardTab()
        self.trades_tab = TradesTab()
        self.market_tab = MarketTab()
        self.analytics_tab = AnalyticsTab()
        self.logs_tab = LogsTab()
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

        state_manager.state_changed.connect(self._on_state)
        state_manager.trade_added.connect(self.trades_tab.add_trade)
        state_manager.log_added.connect(self.logs_tab.add_log)
        state_manager.notification.connect(self._notify)

    def _on_state(self, state: dict) -> None:
        self.dashboard_tab.update_state(state)
        self.market_tab.update_state(state)
        self.analytics_tab.update_state(state)
        self.risk_tab.update_state(state)
        self.system_tab.update_state(state)

    def _notify(self, title: str, message: str) -> None:
        QMessageBox.information(self, title, message)
