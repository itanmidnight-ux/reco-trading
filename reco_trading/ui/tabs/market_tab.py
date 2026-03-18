from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QListWidget, QProgressBar, QVBoxLayout, QWidget

from reco_trading.ui.widgets.stat_card import StatCard


class MarketTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._last_signature: tuple[object, ...] | None = None
        root = QVBoxLayout(self)
        self.header = QLabel("Market Pulse")
        self.header.setObjectName("sectionTitle")
        root.addWidget(self.header)

        subtitle = QLabel("Live overview of trend, volatility, liquidity and execution quality")
        subtitle.setObjectName("metricLabel")
        root.addWidget(subtitle)
        self.market_ribbon = QLabel("Waiting for market pulse and execution telemetry")
        self.market_ribbon.setObjectName("statusRibbon")
        root.addWidget(self.market_ribbon)

        panel = QFrame()
        panel.setObjectName("panelCard")
        root.addWidget(panel)
        layout = QGridLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setVerticalSpacing(10)

        self.cards = {
            "price": StatCard("Price", compact=True),
            "spread": StatCard("Spread", compact=True),
            "spread_ratio": StatCard("Spread %", compact=True),
            "volatility": StatCard("Volatility", compact=True),
            "order_flow": StatCard("Order Flow", compact=True),
            "trend": StatCard("Trend", compact=True),
            "adx": StatCard("ADX", compact=True),
            "market_regime": StatCard("Market Regime", compact=True),
            "atr": StatCard("ATR", compact=True),
        }
        for i, card in enumerate(self.cards.values()):
            layout.addWidget(card, i // 3, i % 3)

        self.sentiment = QLabel("Sentiment: Neutral")
        self.sentiment.setObjectName("smallMetricValue")
        self.activity = QProgressBar()
        self.activity.setRange(0, 100)
        layout.addWidget(self.sentiment, 3, 0, 1, 2)
        layout.addWidget(self.activity, 3, 2)

        self.orderbook = QListWidget()
        self.orderbook.addItems(["Bid/Ask depth waiting data..."])
        self.movers = QListWidget()
        self.movers.addItems(["Top movers unavailable"])
        layout.addWidget(QLabel("Order Book Snapshot"), 4, 0)
        layout.addWidget(QLabel("Market Intelligence"), 4, 1, 1, 2)
        layout.addWidget(self.orderbook, 5, 0)
        layout.addWidget(self.movers, 5, 1, 1, 2)
        layout.setRowStretch(5, 1)

        self._anim = QPropertyAnimation(self.activity, b"value", self)
        self._anim.setDuration(320)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def update_state(self, state: dict) -> None:
        spread = float(state.get("spread", 0) or 0)
        adx = float(state.get("adx", 0) or 0)
        trend = str(state.get("trend", "-"))
        bid = float(state.get("bid", 0) or 0)
        ask = float(state.get("ask", 0) or 0)
        price = float(state.get("current_price", state.get("price", 0)) or 0)
        volume = float(state.get("volume", 0) or 0)
        atr = float(state.get("atr", 0) or 0)
        spread_ratio = (spread / price * 100) if price > 0 else 0.0
        signature = (
            round(price, 8),
            round(spread, 8),
            round(spread_ratio, 8),
            str(state.get("volatility_regime", "-")),
            str(state.get("order_flow", "-")),
            trend,
            round(adx, 8),
            str(state.get("market_regime", "-")),
            round(atr, 8),
            round(bid, 8),
            round(ask, 8),
            round(volume, 8),
            state.get("distance_to_support", "-"),
            state.get("distance_to_resistance", "-"),
        )
        if signature == self._last_signature:
            return
        self._last_signature = signature

        self.cards["price"].set_value(f"{price:.2f}")
        self.cards["spread"].set_value(f"{spread:.6f}")
        self.cards["spread_ratio"].set_value(f"{spread_ratio:.4f}%")
        self.cards["volatility"].set_value(str(state.get("volatility_regime", "-")))
        self.cards["order_flow"].set_value(str(state.get("order_flow", "-")))
        self.cards["trend"].set_value(trend)
        self.cards["adx"].set_value(f"{adx:.2f}")
        self.cards["market_regime"].set_value(str(state.get("market_regime", "-")))
        self.cards["atr"].set_value(f"{atr:.4f}")

        normalized_trend = trend.upper().replace("BUY", "UP").replace("SELL", "DOWN")
        sentiment = "Bullish" if "UP" in normalized_trend else "Bearish" if "DOWN" in normalized_trend else "Neutral"
        self.sentiment.setText(f"Sentiment: {sentiment}")
        self._anim.stop()
        self._anim.setStartValue(self.activity.value())
        self._anim.setEndValue(max(0, min(100, int(adx))))
        self._anim.start()

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
                f"Distance to Support: {state.get('distance_to_support', '-')}",
                f"Distance to Resistance: {state.get('distance_to_resistance', '-')}",
            ]
        )
        self.market_ribbon.setText(
            f"Price {price:.2f} • Spread {spread_ratio:.4f}% • Trend {trend} • Regime {state.get('market_regime', '-')}"
        )
