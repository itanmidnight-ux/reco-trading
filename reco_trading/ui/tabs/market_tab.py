from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QWidget

from reco_trading.ui.widgets.stat_card import StatCard


class MarketTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QGridLayout(self)
        self.cards = {
            "spread": StatCard("Spread", compact=True),
            "volatility": StatCard("Volatility", compact=True),
            "order_flow": StatCard("Order Flow", compact=True),
            "trend": StatCard("Trend Metrics", compact=True),
            "adx": StatCard("ADX", compact=True),
        }
        for i, card in enumerate(self.cards.values()):
            layout.addWidget(card, i // 3, i % 3)

    def update_state(self, state: dict) -> None:
        self.cards["spread"].set_value(str(state.get("spread", "-")))
        self.cards["volatility"].set_value(str(state.get("volatility_regime", "-")))
        self.cards["order_flow"].set_value(str(state.get("order_flow", "-")))
        self.cards["trend"].set_value(str(state.get("trend", "-")))
        self.cards["adx"].set_value(str(state.get("adx", "-")))
