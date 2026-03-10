from __future__ import annotations

from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout, QWidget

from reco_trading.ui.chart_widget import CandlestickChartWidget
from reco_trading.ui.components.formatting import as_text, fmt_number, fmt_price, trend_arrow


class MarketTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        title = QLabel("Market")
        title.setObjectName("sectionTitle")
        root.addWidget(title)

        metrics = QFrame()
        metrics.setObjectName("panelCard")
        grid = QGridLayout(metrics)
        self.labels: dict[str, QLabel] = {}
        for i, key in enumerate(["price", "spread", "order_flow", "volatility", "trend", "adx"]):
            k = QLabel(key.replace("_", " ").title())
            k.setObjectName("metricLabel")
            v = QLabel("--")
            self.labels[key] = v
            grid.addWidget(k, i, 0)
            grid.addWidget(v, i, 1)
        root.addWidget(metrics)

        self.chart = CandlestickChartWidget()
        root.addWidget(self.chart)

    def update_state(self, state: dict) -> None:
        trend = as_text(state.get("trend"))
        self.labels["price"].setText(fmt_price(state.get("current_price", state.get("price", 0))))
        self.labels["spread"].setText(fmt_number(state.get("spread", 0), 4))
        self.labels["order_flow"].setText(as_text(state.get("order_flow")))
        self.labels["volatility"].setText(as_text(state.get("volatility_regime")))
        self.labels["trend"].setText(f"{trend_arrow(trend)} {trend}")
        self.labels["adx"].setText(fmt_number(state.get("adx", 0), 2))
        self.chart.update_from_snapshot(state)
