from __future__ import annotations

from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QListWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from reco_trading.ui.widgets.pnl_chart import PnlChart
from reco_trading.ui.widgets.stat_card import StatCard


class AnalyticsTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        title = QLabel("Performance Analytics")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        subtitle = QLabel("Detailed strategy metrics and equity curve")
        subtitle.setObjectName("metricLabel")
        layout.addWidget(subtitle)

        panel = QFrame()
        panel.setObjectName("panelCard")
        layout.addWidget(panel)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 12, 12, 12)

        grid = QGridLayout()
        self.cards = {}
        keys = ["total_trades", "win_rate", "profit_factor", "average_win", "average_loss"]
        for i, key in enumerate(keys):
            card = StatCard(key.replace("_", " ").title(), compact=True)
            self.cards[key] = card
            grid.addWidget(card, i // 3, i % 3)
        panel_layout.addLayout(grid)

        self.equity_curve = PnlChart("Performance")
        panel_layout.addWidget(self.equity_curve)

        self.analysis_table = QTableWidget(0, 2)
        self.analysis_table.setHorizontalHeaderLabels(["Analysis", "Value"])
        self.analysis_table.horizontalHeader().setStretchLastSection(True)
        panel_layout.addWidget(self.analysis_table)

        self.insights = QListWidget()
        self.insights.addItems(["Waiting analytics signals..."])
        panel_layout.addWidget(self.insights)

    def update_state(self, state: dict) -> None:
        analytics = state.get("analytics", {})
        for key, card in self.cards.items():
            val = analytics.get(key, "-")
            if key == "win_rate":
                try:
                    card.set_value(f"{float(val) * 100:.2f}%")
                except (TypeError, ValueError):
                    card.set_value("-")
            else:
                card.set_value(str(val))
        self.equity_curve.plot([float(v) for v in analytics.get("equity_curve", []) if isinstance(v, (int, float))])

        metric_rows = [
            ("Sharpe Ratio", analytics.get("sharpe_ratio", "-")),
            ("Sortino Ratio", analytics.get("sortino_ratio", "-")),
            ("Max Drawdown", analytics.get("max_drawdown", "-")),
            ("Expectancy", analytics.get("expectancy", "-")),
            ("Avg Duration", analytics.get("avg_trade_duration", "-")),
        ]
        self.analysis_table.setRowCount(0)
        for row, (name, value) in enumerate(metric_rows):
            self.analysis_table.insertRow(row)
            self.analysis_table.setItem(row, 0, QTableWidgetItem(name))
            self.analysis_table.setItem(row, 1, QTableWidgetItem(str(value)))

        self.insights.clear()
        self.insights.addItem(f"Signal: {state.get('signal', '-')}")
        self.insights.addItem(f"Confidence: {state.get('confidence', 0)}")
        self.insights.addItem(f"Open position: {state.get('position_side', state.get('open_position', '-'))}")
        self.insights.addItem(f"Trades today: {state.get('trades_today', 0)}")
