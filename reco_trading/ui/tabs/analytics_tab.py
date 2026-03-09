from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QVBoxLayout, QWidget

from reco_trading.ui.widgets.pnl_chart import PnlChart
from reco_trading.ui.widgets.stat_card import StatCard


class AnalyticsTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        grid = QGridLayout()
        self.cards = {}
        keys = ["total_trades", "win_rate", "profit_factor", "average_win", "average_loss", "largest_win", "largest_loss"]
        for i, key in enumerate(keys):
            card = StatCard(key.replace("_", " ").title())
            self.cards[key] = card
            grid.addWidget(card, i // 4, i % 4)
        layout.addLayout(grid)
        self.equity_curve = PnlChart("Equity curve")
        layout.addWidget(self.equity_curve)

    def update_state(self, state: dict) -> None:
        analytics = state.get("analytics", {})
        for key, card in self.cards.items():
            val = analytics.get(key, "-")
            if isinstance(val, float):
                if key == "win_rate":
                    card.set_value(f"{val:.2%}")
                else:
                    card.set_value(f"{val:.4f}")
            else:
                card.set_value(str(val))
        self.equity_curve.plot(analytics.get("equity_curve", []))
