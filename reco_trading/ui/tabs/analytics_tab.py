from __future__ import annotations

from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout, QWidget

from reco_trading.ui.widgets.pnl_chart import PnlChart
from reco_trading.ui.widgets.stat_card import StatCard


class AnalyticsTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        title = QLabel("Performance Analytics")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        subtitle = QLabel("Detailed strategy metrics and equity curve")
        subtitle.setObjectName("metricLabel")
        layout.addWidget(subtitle)

        panel = QFrame()
        panel.setObjectName("panelCard")
        layout.addWidget(panel)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 12, 12, 12)

        grid = QGridLayout()
        self.cards = {}
        keys = ["total_trades", "win_rate", "profit_factor", "average_win", "average_loss"]
        for i, key in enumerate(keys):
            card = StatCard(key.replace("_", " ").title(), compact=True)
            self.cards[key] = card
            grid.addWidget(card, i // 3, i % 3)
        panel_layout.addLayout(grid)

        self.equity_curve = PnlChart("Performance")
        panel_layout.addWidget(self.equity_curve)

    def update_state(self, state: dict) -> None:
        analytics = state.get("analytics", {})
        for key, card in self.cards.items():
            val = analytics.get(key, "-")
            if key == "win_rate":
                try:
                    card.set_value(f"{float(val) * 100:.2f}%")
                except (TypeError, ValueError):
                    card.set_value("-")
            else:
                card.set_value(str(val))
        self.equity_curve.plot([float(v) for v in analytics.get("equity_curve", []) if isinstance(v, (int, float))])
