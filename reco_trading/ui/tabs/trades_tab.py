from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from reco_trading.ui.widgets.trade_table import TradeTable


class TradeDetailsDialog(QDialog):
    def __init__(self, trade: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Trade {trade.get('trade_id', '-')}")
        self.resize(560, 420)
        form = QFormLayout(self)
        fields = [
            ("trade_id", "Trade ID"),
            ("pair", "Pair"),
            ("side", "Side"),
            ("entry", "Entry Price"),
            ("exit", "Exit Price"),
            ("size", "Position Size"),
            ("pnl", "PnL"),
            ("status", "Status"),
            ("entry_time", "Entry Time"),
            ("exit_time", "Exit Time"),
            ("fees", "Fees"),
            ("confidence", "Confidence"),
            ("signal_details", "Signal Details"),
        ]
        for key, label in fields:
            value = trade.get(key, "-")
            if isinstance(value, (dict, list)):
                details = QTextEdit()
                details.setReadOnly(True)
                details.setMaximumHeight(100)
                details.setPlainText(str(value))
                form.addRow(label, details)
            else:
                form.addRow(label, QLabel(str(value)))


class TradesTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.trade_store: list[dict] = []
        layout = QVBoxLayout(self)
        title = QLabel("Trade Blotter")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        top = QHBoxLayout()
        self.summary = QLabel("0 trades loaded")
        self.summary.setObjectName("metricLabel")
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter by pair, side, status or ID...")
        self.search.textChanged.connect(self._reload_table)
        top.addWidget(self.summary)
        top.addStretch(1)
        top.addWidget(self.search)
        layout.addLayout(top)

        self.table = TradeTable()
        self.table.cellClicked.connect(self._open_detail)
        layout.addWidget(self.table)

    def add_trade(self, trade: dict) -> None:
        self.trade_store.append(trade)
        self._reload_table()

    def update_state(self, state: dict) -> None:
        history = state.get("trade_history", [])
        if len(history) < len(self.trade_store):
            return
        self.trade_store = list(history)
        self._reload_table()

    def _reload_table(self) -> None:
        query = self.search.text().strip().lower()
        trades = self.trade_store
        if query:
            trades = [
                trade
                for trade in trades
                if query in str(trade.get("trade_id", "")).lower()
                or query in str(trade.get("pair", "")).lower()
                or query in str(trade.get("side", "")).lower()
                or query in str(trade.get("status", "")).lower()
            ]
        self.table.load_trades(trades)
        self.summary.setText(f"{len(trades)} trades shown ({len(self.trade_store)} total)")

    def _open_detail(self, row: int, _column: int) -> None:
        trade_id_item: QTableWidgetItem | None = self.table.item(row, 0)
        if not trade_id_item:
            return
        trade = next((t for t in self.trade_store if str(t.get("trade_id")) == trade_id_item.text()), None)
        if trade:
            TradeDetailsDialog(trade, self).exec()
