from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem


class TradeTable(QTableWidget):
    HEADERS = ["Trade ID", "Timestamp", "Symbol", "Side", "Entry price", "Exit price", "Profit/Loss USDT", "Status"]

    def __init__(self) -> None:
        super().__init__(0, len(self.HEADERS))
        self.setHorizontalHeaderLabels(self.HEADERS)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def add_trade(self, trade: dict[str, Any]) -> None:
        row = self.rowCount()
        self.insertRow(row)
        values = [
            str(trade.get("trade_id", "-")),
            str(trade.get("time", trade.get("entry_time", "-"))),
            str(trade.get("pair", "-")),
            str(trade.get("side", "-")),
            _fmt_num(trade.get("entry"), 4),
            _fmt_num(trade.get("exit"), 4) if trade.get("exit") is not None else "-",
            _fmt_num(trade.get("pnl"), 4) if trade.get("pnl") is not None else "-",
            str(trade.get("status", "OPEN")),
        ]
        for col, value in enumerate(values):
            self.setItem(row, col, QTableWidgetItem(value))


def _fmt_num(value: Any, digits: int) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"
