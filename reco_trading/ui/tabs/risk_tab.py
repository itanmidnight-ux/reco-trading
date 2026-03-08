from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QWidget

from reco_trading.ui.widgets.stat_card import StatCard


class RiskTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QGridLayout(self)
        self.cards = {}
        keys = ["risk_per_trade", "max_trades_per_hour", "cooldown", "consecutive_losses", "current_drawdown", "daily_exposure"]
        for i, key in enumerate(keys):
            card = StatCard(key.replace("_", " ").title())
            self.cards[key] = card
            layout.addWidget(card, i // 3, i % 3)

    def update_state(self, state: dict) -> None:
        metrics = state.get("risk_metrics", {})
        for key, card in self.cards.items():
            card.set_value(str(metrics.get(key, "-")))
