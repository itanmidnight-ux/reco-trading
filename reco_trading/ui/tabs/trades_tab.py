from __future__ import annotations

from PySide6.QtWidgets import QLabel, QLineEdit, QVBoxLayout, QWidget

from reco_trading.ui.widgets.trade_table import TradeTable


class TradesTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.trade_store: list[dict] = []

        layout = QVBoxLayout(self)
        title = QLabel("Trades")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter by trade_id, side, timestamp...")
        self.filter.textChanged.connect(self._reload)
        layout.addWidget(self.filter)

        self.table = TradeTable()
        layout.addWidget(self.table)

    def update_state(self, state: dict) -> None:
        self.trade_store = list(state.get("trade_history", []))
        self._reload()

    def _reload(self) -> None:
        q = self.filter.text().strip().lower()
        trades = self.trade_store
        if q:
            trades = [t for t in trades if q in str(t).lower()]
        self.table.load_trades(trades)
