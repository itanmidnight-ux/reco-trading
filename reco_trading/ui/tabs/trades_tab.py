from __future__ import annotations

from PySide6.QtWidgets import QDialog, QFormLayout, QLabel, QTableWidgetItem, QVBoxLayout, QWidget

from reco_trading.ui.widgets.trade_table import TradeTable


class TradeDetailsDialog(QDialog):
    def __init__(self, trade: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Trade {trade.get('trade_id', '-')}")
        form = QFormLayout(self)
        fields = [
            ("entry_time", "Entry time"),
            ("exit_time", "Exit time"),
            ("size", "Size"),
            ("fees", "Fees"),
            ("pnl", "Net PnL"),
            ("confidence", "Confidence score"),
            ("signal_details", "Signal details"),
        ]
        for key, label in fields:
            form.addRow(label, QLabel(str(trade.get(key, "-"))))


class TradesTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.trade_store: list[dict] = []
        layout = QVBoxLayout(self)
        self.table = TradeTable()
        self.table.cellDoubleClicked.connect(self._open_detail)
        layout.addWidget(self.table)

    def add_trade(self, trade: dict) -> None:
        self.trade_store.append(trade)
        self.table.add_trade(trade)

    def _open_detail(self, row: int, _column: int) -> None:
        trade_id_item: QTableWidgetItem | None = self.table.item(row, 0)
        if not trade_id_item:
            return
        trade_id = trade_id_item.text()
        trade = next((t for t in self.trade_store if str(t.get("trade_id")) == trade_id), None)
        if not trade:
            return
        TradeDetailsDialog(trade, self).exec()
