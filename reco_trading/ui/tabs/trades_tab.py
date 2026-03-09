from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget

from reco_trading.ui.widgets.trade_table import TradeTable


class TradesTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.trade_store: list[dict] = []

        layout = QVBoxLayout(self)
        title = QLabel("Trades")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        top = QHBoxLayout()
        self.summary = QLabel("0 trades loaded")
        self.summary.setObjectName("metricLabel")
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter by side, time or price…")
        self.search.textChanged.connect(self._reload_table)
        top.addWidget(self.summary)
        top.addStretch(1)
        top.addWidget(self.search)
        layout.addLayout(top)

        self.table = TradeTable()
        layout.addWidget(self.table, 1)

    def add_trade(self, trade: dict) -> None:
        self.trade_store.append(trade or {})
        self._reload_table()

    def update_state(self, state: dict) -> None:
        try:
            self.trade_store = list(state.get("trade_history", []) or [])
            self._reload_table()
        except Exception:
            return

    def _reload_table(self) -> None:
        query = self.search.text().strip().lower()
        trades = self.trade_store
        if query:
            trades = [
                trade
                for trade in self.trade_store
                if query in str(trade.get("side", "")).lower()
                or query in str(trade.get("time", trade.get("entry_time", ""))).lower()
                or query in str(trade.get("price", trade.get("entry_price", ""))).lower()
            ]
        self.table.load_trades(trades)
        self.summary.setText(f"{len(trades)} trades shown ({len(self.trade_store)} total)")
