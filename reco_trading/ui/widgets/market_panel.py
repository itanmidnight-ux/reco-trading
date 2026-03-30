from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QGridLayout, QWidget

from reco_trading.ui.widgets.stat_card import StatCard


class MarketPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QGridLayout(self)
        self.cards: dict[str, StatCard] = {}
        keys = [
            ("current_price", "Current price"),
            ("change_24h", "24h change %"),
            ("spread", "Spread"),
            ("volume", "Volume"),
            ("volatility_regime", "Volatility regime"),
        ]
        for i, (key, label) in enumerate(keys):
            card = StatCard(label)
            layout.addWidget(card, i // 3, i % 3)
            self.cards[key] = card

    def update_market(self, state: dict[str, Any]) -> None:
        self.cards["current_price"].set_value(_fmt_num(state.get("current_price"), 4))
        self.cards["change_24h"].set_value(f"{_fmt_num(state.get('change_24h'), 2)}%")
        self.cards["spread"].set_value(_fmt_num(state.get("spread"), 6))
        self.cards["volume"].set_value(_fmt_num(state.get("volume"), 2))
        self.cards["volatility_regime"].set_value(str(state.get("volatility_regime") or "-"))


def _fmt_num(value: Any, digits: int) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"
