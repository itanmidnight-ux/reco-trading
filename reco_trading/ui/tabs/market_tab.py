from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QListWidget, QProgressBar, QVBoxLayout, QWidget

from reco_trading.ui.chart_widget import CandlestickChartWidget
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

        content = QHBoxLayout()
        content.setSpacing(10)
        root.addLayout(content, 1)

        self.left_panel = QFrame()
        self.left_panel.setObjectName("panelCard")
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.addWidget(self._section_title("Live Market Stats"))
        self.cards = {
            "price": StatCard("Price", compact=True),
            "spread": StatCard("Spread", compact=True),
            "spread_ratio": StatCard("Spread %", compact=True),
            "trend": StatCard("Trend", compact=True),
            "adx": StatCard("ADX", compact=True),
            "atr": StatCard("ATR", compact=True),
        }
        for card in self.cards.values():
            left_layout.addWidget(card)
        self.sentiment = QLabel("Sentiment: Neutral")
        self.sentiment.setObjectName("smallMetricValue")
        self.activity = QProgressBar()
        self.activity.setRange(0, 100)
        left_layout.addWidget(self.sentiment)
        left_layout.addWidget(self.activity)
        left_layout.addStretch(1)

        self.chart_panel = QFrame()
        self.chart_panel.setObjectName("panelCard")
        chart_layout = QVBoxLayout(self.chart_panel)
        chart_layout.setContentsMargins(12, 12, 12, 12)
        chart_layout.addWidget(self._section_title("Expanded Market Chart"))
        self.chart = CandlestickChartWidget()
        chart_layout.addWidget(self.chart)

        self.right_panel = QFrame()
        self.right_panel.setObjectName("panelCard")
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.addWidget(self._section_title("Execution Context"))
        self.context_cards = {
            "volatility": StatCard("Volatility", compact=True),
            "order_flow": StatCard("Order Flow", compact=True),
            "market_regime": StatCard("Market Regime", compact=True),
        }
        for card in self.context_cards.values():
            right_layout.addWidget(card)
        self.orderbook = QListWidget()
        self.orderbook.addItems(["Bid/Ask depth waiting data..."])
        self.movers = QListWidget()
        self.movers.addItems(["Top movers unavailable"])
        right_layout.addWidget(self._section_title("Order Book Snapshot"))
        right_layout.addWidget(self.orderbook)
        right_layout.addWidget(self._section_title("Market Intelligence"))
        right_layout.addWidget(self.movers)

        content.addWidget(self.left_panel, 2)
        content.addWidget(self.chart_panel, 5)
        content.addWidget(self.right_panel, 2)

    def _section_title(self, title: str) -> QLabel:
        label = QLabel(title)
        label.setObjectName("metricLabel")
        return label

    def _section_title(self, title: str) -> QLabel:
        label = QLabel(title)
        label.setObjectName("metricLabel")
        return label

    def update_state(self, state: dict) -> None:
        spread = float(state.get("spread", 0) or 0)
        adx = float(state.get("adx", 0) or 0)
        trend = str(state.get("trend", "-"))
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
        self.cards["trend"].set_value(trend)
        self.cards["adx"].set_value(f"{adx:.2f}")
        self.cards["atr"].set_value(f"{atr:.4f}")
        self.context_cards["volatility"].set_value(str(state.get("volatility_regime", "-")))
        self.context_cards["order_flow"].set_value(str(state.get("order_flow", "-")))
        self.context_cards["market_regime"].set_value(str(state.get("market_regime", "-")))

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
        self.chart.update_from_snapshot(state)
        self.market_footer.setText(
            f"Volatility {state.get('volatility_regime', '-')} • Order flow {state.get('order_flow', '-')} • "
            f"Volume {volume:.2f} • Support {state.get('distance_to_support', '-')} • "
            f"Resistance {state.get('distance_to_resistance', '-')}"
        )
        self.market_ribbon.setText(
            f"Price {price:.2f} • Spread {spread_ratio:.4f}% • Trend {trend} • Regime {state.get('market_regime', '-')}"
        )
