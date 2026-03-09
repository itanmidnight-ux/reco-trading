from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QListWidget, QProgressBar, QVBoxLayout, QWidget

from reco_trading.ui.widgets.stat_card import StatCard


class MarketTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        self.header = QLabel("Market Pulse")
        self.header.setObjectName("sectionTitle")
        root.addWidget(self.header)

        subtitle = QLabel("Live overview of trend, volatility and order flow")
        subtitle.setObjectName("metricLabel")
        root.addWidget(subtitle)

        panel = QFrame()
        panel.setObjectName("panelCard")
        root.addWidget(panel)
        layout = QGridLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setVerticalSpacing(10)

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

        self.orderbook = QListWidget()
        self.orderbook.addItems(["Bid/Ask depth waiting data..."])
        self.movers = QListWidget()
        self.movers.addItems(["Top movers unavailable"])
        layout.addWidget(QLabel("Order Book Snapshot"), 3, 0)
        layout.addWidget(QLabel("Price Action Highlights"), 3, 1, 1, 2)
        layout.addWidget(self.orderbook, 4, 0)
        layout.addWidget(self.movers, 4, 1, 1, 2)
        layout.setRowStretch(4, 1)

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

        bid = float(state.get("bid", 0) or 0)
        ask = float(state.get("ask", 0) or 0)
        price = float(state.get("current_price", state.get("price", 0)) or 0)
        volume = float(state.get("volume", 0) or 0)
        atr = float(state.get("atr", 0) or 0)
        self.orderbook.clear()
        self.orderbook.addItems(
            [
                f"Best Bid: {bid:.2f}",
                f"Best Ask: {ask:.2f}",
                f"Spread (abs): {(ask - bid):.6f}",
                f"Price: {price:.2f}",
            ]
        )

        self.movers.clear()
        self.movers.addItems(
            [
                f"ATR: {atr:.4f}",
                f"Volatility Regime: {state.get('volatility_regime', '-')}",
                f"Order Flow Bias: {state.get('order_flow', '-')}",
                f"Volume: {volume:.2f}",
            ]
        )
