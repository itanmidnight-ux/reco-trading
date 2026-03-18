from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from reco_trading.ui.widgets.stat_card import StatCard
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
        self._history_signature: tuple[tuple[str, str, str, str], ...] = tuple()
        layout = QVBoxLayout(self)
        title = QLabel("Trade Blotter")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        summary_grid = QGridLayout()
        self.summary_cards = {
            "total": StatCard("Total Trades", compact=True),
            "open": StatCard("Open Trades", compact=True),
            "closed": StatCard("Closed Trades", compact=True),
            "realized_pnl": StatCard("Realized PnL", compact=True),
            "win_rate": StatCard("Win Rate", compact=True),
        }
        for i, card in enumerate(self.summary_cards.values()):
            summary_grid.addWidget(card, 0, i)
        layout.addLayout(summary_grid)

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
        self._history_signature = self._signature_for(self.trade_store)
        self._reload_table()

    def update_state(self, state: dict) -> None:
        history = state.get("trade_history", [])
        signature = self._signature_for(history)
        if signature == self._history_signature:
            return
        if len(history) < len(self.trade_store):
            self._history_signature = signature
            self.trade_store = list(history)
            self._reload_table()
            return
        self._history_signature = signature
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
        self._update_kpis()

    def _update_kpis(self) -> None:
        total = len(self.trade_store)
        open_count = sum(1 for t in self.trade_store if str(t.get("status", "")).upper() == "OPEN")
        closed = total - open_count

        pnl_values = [float(t.get("pnl", 0) or 0) for t in self.trade_store if t.get("pnl") not in {None, "-"}]
        realized_pnl = sum(pnl_values)
        wins = sum(1 for p in pnl_values if p > 0)
        win_rate = (wins / len(pnl_values) * 100) if pnl_values else 0.0

        self.summary_cards["total"].set_value(str(total))
        self.summary_cards["open"].set_value(str(open_count))
        self.summary_cards["closed"].set_value(str(closed))
        self.summary_cards["realized_pnl"].set_value(f"{realized_pnl:.4f}")
        self.summary_cards["win_rate"].set_value(f"{win_rate:.1f}%")

    def _open_detail(self, row: int, _column: int) -> None:
        trade_id_item: QTableWidgetItem | None = self.table.item(row, 0)
        if not trade_id_item:
            return
        trade = next((t for t in self.trade_store if str(t.get("trade_id")) == trade_id_item.text()), None)
        if trade:
            TradeDetailsDialog(trade, self).exec()

    def _signature_for(self, trades: list[dict]) -> tuple[tuple[str, str, str, str], ...]:
        return tuple(
            (
                str(trade.get("trade_id", "")),
                str(trade.get("status", "")),
                str(trade.get("pnl", "")),
                str(trade.get("exit_time", "")),
            )
            for trade in trades
        )
