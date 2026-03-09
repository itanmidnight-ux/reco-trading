from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QWidget

from reco_trading.ui.widgets.stat_card import StatCard


class MarketTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QGridLayout(self)
        fields = [
            ("spread", "Spread"),
            ("volatility_regime", "Volatility"),
            ("order_flow", "Order Flow"),
            ("trend", "Trend"),
            ("adx", "ADX"),
            ("volume", "Volume"),
        ]
        self.cards: dict[str, StatCard] = {}
        for i, (key, label) in enumerate(fields):
            card = StatCard(label)
            self.cards[key] = card
            layout.addWidget(card, i // 3, i % 3)

    def update_state(self, state: dict) -> None:
        for k, c in self.cards.items():
            c.set_value(str(state.get(k, "-")))
