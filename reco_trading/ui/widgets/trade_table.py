from __future__ import annotations

from typing import Any

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem


class TradeTable(QTableWidget):
    HEADERS = ["Trade ID", "Pair", "Entry Price", "Exit Price", "Position Size", "PnL", "Status", "Timestamp"]

    def __init__(self) -> None:
        super().__init__(0, len(self.HEADERS))
        self.setHorizontalHeaderLabels(self.HEADERS)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setAlternatingRowColors(True)

    def add_trade(self, trade: dict[str, Any]) -> None:
        row = self.rowCount()
        self.insertRow(row)
        pnl = _to_float(trade.get("pnl"))
        values = [
            str(trade.get("trade_id", "-")),
            str(trade.get("pair", "BTC/USDT")),
            _fmt_num(trade.get("entry"), 4),
            _fmt_num(trade.get("exit"), 4) if trade.get("exit") is not None else "-",
            _fmt_num(trade.get("size"), 4),
            _fmt_num(trade.get("pnl"), 4),
            str(trade.get("status", "OPEN")),
            str(trade.get("time", trade.get("entry_time", "-"))),
        ]
        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            if pnl is not None:
                item.setForeground(QColor("#16c784") if pnl >= 0 else QColor("#ea3943"))
            self.setItem(row, col, item)


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_num(value: Any, digits: int) -> str:
    n = _to_float(value)
    return "-" if n is None else f"{n:.{digits}f}"
