from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QWidget

from reco_trading.ui.widgets.stat_card import StatCard


class DashboardTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QGridLayout(self)
        self.cards: dict[str, StatCard] = {}
        labels = [
            ("pair", "Pair"), ("timeframe", "Timeframe"), ("current_price", "Current price"), ("spread", "Spread"),
            ("trend", "Trend"), ("adx", "ADX"), ("volatility_regime", "Volatility regime"), ("order_flow", "Order flow"),
            ("signal", "Signal"), ("confidence", "Confidence"), ("balance", "Balance"), ("equity", "Equity"),
            ("daily_pnl", "Daily PnL"), ("trades_today", "Trades today"), ("win_rate", "Win rate"), ("status", "RUNNING / PAUSED"),
            ("last_trade", "Last trade"), ("cooldown", "Cooldown timer"),
        ]
        for idx, (key, label) in enumerate(labels):
            card = StatCard(label)
            self.cards[key] = card
            layout.addWidget(card, idx // 3, idx % 3)

    def update_state(self, state: dict) -> None:
        for key, card in self.cards.items():
            value = state.get(key, "-")
            if isinstance(value, float):
                if key == "confidence" or key == "win_rate":
                    card.set_value(f"{value:.2%}")
                else:
                    card.set_value(f"{value:.4f}")
            else:
                card.set_value(str(value))
