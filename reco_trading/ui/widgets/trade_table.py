from __future__ import annotations

from typing import Any

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem


class TradeTable(QTableWidget):
    HEADERS = ["Trade ID", "Pair", "Entry Price", "Exit Price", "Position Size", "PnL", "Status", "Timestamp"]

    def __init__(self) -> None:
        super().__init__(0, len(self.HEADERS))
        self.setHorizontalHeaderLabels(self.HEADERS)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setAlternatingRowColors(True)

    def add_trade(self, trade: dict[str, Any]) -> None:
        row = self.rowCount()
        self.insertRow(row)
        pnl = trade.get("pnl")
        values = [
            str(trade.get("trade_id", "-")),
            str(trade.get("pair", "BTC/USDT")),
            _fmt_num(trade.get("entry", trade.get("entry_price")), 4),
            _fmt_num(trade.get("exit", trade.get("exit_price")), 4),
            _fmt_num(trade.get("size", trade.get("position_size")), 4),
            _fmt_num(pnl, 4),
            str(trade.get("status", "OPEN")),
            str(trade.get("time", trade.get("entry_time", "-"))),
        ]
        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            if col == 5:
                item.setForeground(QColor("#16c784" if _to_float(pnl) >= 0 else "#ea3943"))
            self.setItem(row, col, item)


def _fmt_num(value: Any, digits: int) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
