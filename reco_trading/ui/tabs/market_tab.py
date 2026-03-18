from __future__ import annotations

from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout, QWidget

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

        self.metrics_panel = QFrame()
        self.metrics_panel.setObjectName("panelCard")
        metrics_layout = QGridLayout(self.metrics_panel)
        metrics_layout.setContentsMargins(12, 12, 12, 12)
        metrics_layout.setSpacing(8)
        self.cards = {
            "price": StatCard("Price", compact=True),
            "spread": StatCard("Spread", compact=True),
            "trend": StatCard("Trend", compact=True),
            "adx": StatCard("ADX", compact=True),
            "regime": StatCard("Regime", compact=True),
            "atr": StatCard("ATR", compact=True),
        }
        for i, card in enumerate(self.cards.values()):
            metrics_layout.addWidget(card, 0, i)
        root.addWidget(self.metrics_panel)

        self.chart_panel = QFrame()
        self.chart_panel.setObjectName("panelCard")
        self.chart_panel.setMinimumHeight(620)
        chart_layout = QVBoxLayout(self.chart_panel)
        chart_layout.setContentsMargins(12, 12, 12, 12)
        chart_layout.addWidget(self._section_title("Expanded Market Chart"))
        self.chart = CandlestickChartWidget()
        chart_layout.addWidget(self.chart)
        self.market_footer = QLabel("Waiting for live market context")
        self.market_footer.setObjectName("smallMetricValue")
        self.market_footer.setWordWrap(True)
        chart_layout.addWidget(self.market_footer)
        root.addWidget(self.chart_panel, 1)

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
        candles = state.get("candles_5m", [])[-2:]
        signature = (
            round(price, 8),
            round(spread, 8),
            round(spread_ratio, 8),
            str(state.get("volatility_regime", "-")),
            trend,
            round(adx, 8),
            str(state.get("market_regime", "-")),
            round(atr, 8),
            round(volume, 8),
            state.get("distance_to_support", "-"),
            state.get("distance_to_resistance", "-"),
            tuple(
                (
                    round(float(c.get("open", 0.0)), 8),
                    round(float(c.get("high", 0.0)), 8),
                    round(float(c.get("low", 0.0)), 8),
                    round(float(c.get("close", 0.0)), 8),
                    round(float(c.get("volume", 0.0)), 8),
                )
                for c in candles
            ),
        )
        if signature == self._last_signature:
            return
        self._last_signature = signature

        self.cards["price"].set_value(f"{price:.2f}")
        self.cards["spread"].set_value(f"{spread:.6f}")
        self.cards["trend"].set_value(trend)
        self.cards["adx"].set_value(f"{adx:.2f}")
        self.cards["regime"].set_value(str(state.get("market_regime", "-")))
        self.cards["atr"].set_value(f"{atr:.4f}")
        self.market_ribbon.setText(
            f"Price {price:.2f} • Spread {spread_ratio:.4f}% • Trend {trend} • Regime {state.get('market_regime', '-')}"
        )
        self.chart.update_from_snapshot(state)
        self.market_footer.setText(
            f"Volatility {state.get('volatility_regime', '-')} • Order flow {state.get('order_flow', '-')} • "
            f"Volume {volume:.2f} • Support {state.get('distance_to_support', '-')} • "
            f"Resistance {state.get('distance_to_resistance', '-')}"
        )
