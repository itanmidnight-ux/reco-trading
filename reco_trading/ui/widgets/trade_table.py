from __future__ import annotations

from typing import Any

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem


class TradeTable(QTableWidget):
    HEADERS = ["Time", "Side", "Price", "Quantity", "Value", "PnL"]

    def __init__(self) -> None:
        super().__init__(0, len(self.HEADERS))
        self.setHorizontalHeaderLabels(self.HEADERS)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)

    def load_trades(self, trades: list[dict[str, Any]]) -> None:
        self.setRowCount(0)
        for trade in trades:
            self.add_trade(trade)

    def add_trade(self, trade: dict[str, Any]) -> None:
        row = self.rowCount()
        self.insertRow(row)
        side = str(trade.get("side", trade.get("signal", "-"))).upper()
        qty = _to_float(trade.get("quantity", trade.get("size", trade.get("position_size"))))
        price = _to_float(trade.get("price", trade.get("entry", trade.get("entry_price"))))
        pnl = _to_float(trade.get("pnl"), None)
        value = (qty * price) if qty is not None and price is not None else None

        values = [
            str(trade.get("time", trade.get("entry_time", "-"))),
            side if side else "-",
            _fmt_num(price, 4),
            _fmt_num(qty, 6),
            _fmt_num(value, 2),
            _fmt_num(pnl, 2),
        ]
        for col, value_text in enumerate(values):
            item = QTableWidgetItem(value_text)
            if col == 1:
                item.setForeground(QColor("#16c784" if side == "BUY" else "#ea3943" if side == "SELL" else "#e6e8ee"))
            if col == 5 and pnl is not None:
                item.setForeground(QColor("#16c784" if pnl >= 0 else "#ea3943"))
            self.setItem(row, col, item)


def _to_float(value: Any, fallback: float | None = 0.0) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _fmt_num(value: Any, digits: int) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"
