from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QWidget

from reco_trading.ui.widgets.stat_card import StatCard


class MarketPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QGridLayout(self)
        self.cards = {}
        keys = ["Price", "Bid", "Ask", "Spread", "Volume", "ATR", "ADX", "Volatility"]
        for i, key in enumerate(keys):
            card = StatCard(key)
            layout.addWidget(card, i // 4, i % 4)
            self.cards[key.lower()] = card

    def update_market(self, state: dict) -> None:
        self.cards["price"].set_value(f"{state.get('current_price', 0.0):.4f}")
        self.cards["bid"].set_value(f"{state.get('bid', 0.0):.4f}")
        self.cards["ask"].set_value(f"{state.get('ask', 0.0):.4f}")
        self.cards["spread"].set_value(f"{state.get('spread', 0.0):.6f}")
        self.cards["volume"].set_value(f"{state.get('volume', 0.0):.2f}")
        self.cards["atr"].set_value(f"{state.get('atr', 0.0):.4f}")
        self.cards["adx"].set_value(f"{state.get('adx', 0.0):.2f}")
        self.cards["volatility"].set_value(str(state.get("volatility_regime", "-")))
