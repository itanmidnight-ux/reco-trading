from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QMainWindow, QStackedWidget, QVBoxLayout, QWidget

from reco_trading.ui.components.navigation import Sidebar
from reco_trading.ui.components.formatting import as_text, fmt_number, fmt_price
from reco_trading.ui.state_manager import StateManager
from reco_trading.ui.tabs.analytics_tab import AnalyticsTab
from reco_trading.ui.tabs.dashboard_tab import DashboardTab
from reco_trading.ui.tabs.logs_tab import LogsTab
from reco_trading.ui.tabs.market_tab import MarketTab
from reco_trading.ui.tabs.risk_tab import RiskTab
from reco_trading.ui.tabs.settings_tab import SettingsTab
from reco_trading.ui.tabs.system_tab import SystemTab
from reco_trading.ui.tabs.trades_tab import TradesTab
from reco_trading.ui.theme import app_stylesheet


class MainWindow(QMainWindow):
    """Terminal-like monitoring UI that reads engine snapshots only."""

    TAB_ITEMS = [
        ("📊", "Dashboard"),
        ("🧾", "Trades"),
        ("🌐", "Market"),
        ("📈", "Analytics"),
        ("📜", "Logs"),
        ("🛡", "Risk"),
        ("⚙", "Settings"),
        ("🖥", "System"),
    ]

    def __init__(self, state_manager: StateManager) -> None:
        super().__init__()
        self.state_manager = state_manager
        self.setWindowTitle("Reco Trading Terminal")
        self.resize(1680, 980)
        self.setStyleSheet(app_stylesheet())

        root = QWidget()
        root_layout = QHBoxLayout(root)

        self.sidebar = Sidebar(self.TAB_ITEMS)
        root_layout.addWidget(self.sidebar)

        content = QVBoxLayout()
        self.top_bar = self._build_top_bar()
        content.addWidget(self.top_bar)

        self.stack = QStackedWidget()
        self.dashboard_tab = DashboardTab(state_manager=state_manager)
        self.trades_tab = TradesTab()
        self.market_tab = MarketTab()
        self.analytics_tab = AnalyticsTab()
        self.logs_tab = LogsTab()
        self.risk_tab = RiskTab()
        self.settings_tab = SettingsTab()
        self.system_tab = SystemTab()
        for tab in [
            self.dashboard_tab,
            self.trades_tab,
            self.market_tab,
            self.analytics_tab,
            self.logs_tab,
            self.risk_tab,
            self.settings_tab,
            self.system_tab,
        ]:
            self.stack.addWidget(tab)
        content.addWidget(self.stack)

        root_layout.addLayout(content, stretch=1)
        self.setCentralWidget(root)

        self.sidebar.tab_selected.connect(self._set_tab)
        self.state_manager.state_changed.connect(self._on_state)

    def _build_top_bar(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("topBar")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 8, 12, 8)

        self.top_pair = QLabel("BTC/USDT")
        self.top_pair.setObjectName("metricValue")
        self.top_status = QLabel("State: --")
        self.top_conn = QLabel("Conn: --")
        self.top_price = QLabel("Price: --")
        self.top_spread = QLabel("Spread: --")
        self.top_vol = QLabel("Vol: --")

        for widget in [self.top_pair, self.top_status, self.top_conn, self.top_price, self.top_spread, self.top_vol]:
            layout.addWidget(widget)
            layout.addSpacing(16)
        layout.addStretch(1)
        return frame

    def _set_tab(self, index: int) -> None:
        self.sidebar.set_active(index)
        self.stack.setCurrentIndex(index)
        self._animate_tab_change()

    def _animate_tab_change(self) -> None:
        current = self.stack.currentWidget()
        if not current:
            return
        anim = QPropertyAnimation(current, b"windowOpacity", self)
        anim.setDuration(180)
        anim.setStartValue(0.6)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.start()
        self._anim = anim

    def _on_state(self, state: dict) -> None:
        self.top_pair.setText(as_text(state.get("pair"), "BTC/USDT"))
        self.top_status.setText(f"State: {as_text(state.get('bot_state') or state.get('status'))}")
        system = state.get("system", {}) if isinstance(state.get("system"), dict) else {}
        self.top_conn.setText(f"Conn: {as_text(system.get('exchange_status', 'UNKNOWN'))}")
        self.top_price.setText(f"Price: {fmt_price(state.get('current_price', state.get('price', 0)))}")
        self.top_spread.setText(f"Spread: {fmt_number(state.get('spread', 0), 4)}")
        self.top_vol.setText(f"Vol: {as_text(state.get('volatility_regime'))}")

        for tab in [self.dashboard_tab, self.trades_tab, self.market_tab, self.analytics_tab, self.logs_tab, self.risk_tab, self.system_tab, self.settings_tab]:
            if hasattr(tab, "update_state"):
                tab.update_state(state)
