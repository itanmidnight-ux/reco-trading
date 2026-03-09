from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

from reco_trading.ui.widgets.stat_card import StatCard


class MarketTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        self.header = QLabel("Market Pulse")
        self.header.setObjectName("sectionTitle")
        root.addWidget(self.header)

        panel = QFrame()
        panel.setObjectName("panelCard")
        root.addWidget(panel)
        layout = QGridLayout(panel)

        self.cards = {
            "spread": StatCard("Spread", compact=True),
            "volatility": StatCard("Volatility", compact=True),
            "order_flow": StatCard("Order Flow", compact=True),
            "trend": StatCard("Trend Metrics", compact=True),
            "adx": StatCard("ADX", compact=True),
        }
        for i, card in enumerate(self.cards.values()):
            layout.addWidget(card, i // 3, i % 3)

        self.sentiment = QLabel("Sentiment: Neutral")
        self.sentiment.setObjectName("smallMetricValue")
        self.activity = QProgressBar()
        self.activity.setRange(0, 100)
        layout.addWidget(self.sentiment, 2, 0, 1, 2)
        layout.addWidget(self.activity, 2, 2)

        self._anim = QPropertyAnimation(self.activity, b"value", self)
        self._anim.setDuration(320)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def update_state(self, state: dict) -> None:
        spread = float(state.get("spread", 0) or 0)
        adx = float(state.get("adx", 0) or 0)
        trend = str(state.get("trend", "-"))
        self.cards["spread"].set_value(f"{spread:.6f}")
        self.cards["volatility"].set_value(str(state.get("volatility_regime", "-")))
        self.cards["order_flow"].set_value(str(state.get("order_flow", "-")))
        self.cards["trend"].set_value(trend)
        self.cards["adx"].set_value(f"{adx:.2f}")

        sentiment = "Bullish" if "UP" in trend.upper() else "Bearish" if "DOWN" in trend.upper() else "Neutral"
        self.sentiment.setText(f"Sentiment: {sentiment}")
        self._anim.stop()
        self._anim.setStartValue(self.activity.value())
        self._anim.setEndValue(max(0, min(100, int(adx))))
        self._anim.start()
