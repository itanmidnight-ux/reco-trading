from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QGridLayout, QWidget

from reco_trading.ui.widgets.stat_card import StatCard


class MarketPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QGridLayout(self)
        self.cards: dict[str, StatCard] = {}
        keys = ["Price", "Bid", "Ask", "Spread", "Volume", "ATR", "ADX", "Volatility"]
        for i, key in enumerate(keys):
            card = StatCard(key)
            layout.addWidget(card, i // 4, i % 4)
            self.cards[key.lower()] = card

    def update_market(self, state: dict[str, Any]) -> None:
        self.cards["price"].set_value(_fmt_num(state.get("current_price"), 4))
        self.cards["bid"].set_value(_fmt_num(state.get("bid"), 4))
        self.cards["ask"].set_value(_fmt_num(state.get("ask"), 4))
        self.cards["spread"].set_value(_fmt_num(state.get("spread"), 6))
        self.cards["volume"].set_value(_fmt_num(state.get("volume"), 2))
        self.cards["atr"].set_value(_fmt_num(state.get("atr"), 4))
        self.cards["adx"].set_value(_fmt_num(state.get("adx"), 2))
        self.cards["volatility"].set_value(str(state.get("volatility_regime") or "-"))


def _fmt_num(value: Any, digits: int) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"
