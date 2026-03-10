from __future__ import annotations

from typing import Any

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem


class TradeTable(QTableWidget):
    HEADERS = ["trade_id", "timestamp", "side", "entry_price", "exit_price", "profit_loss", "duration"]

    def __init__(self) -> None:
        super().__init__(0, len(self.HEADERS))
        self.setHorizontalHeaderLabels(self.HEADERS)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setSortingEnabled(True)
        self.setAlternatingRowColors(True)

    def load_trades(self, trades: list[dict[str, Any]]) -> None:
        self.setSortingEnabled(False)
        self.setRowCount(0)
        for trade in trades:
            self._add_trade(trade)
        self.setSortingEnabled(True)

    def _add_trade(self, trade: dict[str, Any]) -> None:
        row = self.rowCount()
        self.insertRow(row)
        pnl = _to_float(trade.get("profit_loss", trade.get("pnl", 0)))
        values = [
            str(trade.get("trade_id", "--")),
            str(trade.get("timestamp", trade.get("time", "--"))),
            str(trade.get("side", trade.get("position_side", "--"))),
            _fmt(trade.get("entry_price", trade.get("entry"))),
            _fmt(trade.get("exit_price", trade.get("exit"))),
            _fmt(pnl),
            str(trade.get("duration", "--")),
        ]
        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            if self.HEADERS[col] == "profit_loss":
                item.setForeground(QColor("#22c55e" if pnl >= 0 else "#ef4444"))
            self.setItem(row, col, item)


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "--"


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
