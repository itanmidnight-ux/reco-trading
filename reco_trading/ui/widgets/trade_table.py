from __future__ import annotations

from PySide6.QtWidgets import QTableWidget, QTableWidgetItem


class TradeTable(QTableWidget):
    HEADERS = ["Trade ID", "Time", "Pair", "Side", "Entry", "Exit", "Size", "PnL (USDT)", "Status"]

    def __init__(self) -> None:
        super().__init__(0, len(self.HEADERS))
        self.setHorizontalHeaderLabels(self.HEADERS)

    def add_trade(self, trade: dict) -> None:
        row = self.rowCount()
        self.insertRow(row)
        values = [
            str(trade.get("trade_id", "-")),
            trade.get("time", "-"),
            trade.get("pair", "-"),
            trade.get("side", "-"),
            f"{trade.get('entry', 0.0):.4f}",
            f"{trade.get('exit', 0.0):.4f}" if trade.get("exit") else "-",
            f"{trade.get('size', 0.0):.6f}",
            f"{trade.get('pnl', 0.0):.4f}" if trade.get("pnl") is not None else "-",
            trade.get("status", "OPEN"),
        ]
        for col, value in enumerate(values):
            self.setItem(row, col, QTableWidgetItem(value))
