from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget

from reco_trading.ui.widgets.pnl_chart import PnlChart


class AnalyticsTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        title = QLabel("Analytics")
        title.setObjectName("sectionTitle")
        root.addWidget(title)

        grid = QGridLayout()
        self.equity = PnlChart("Equity Curve")
        self.pnl = PnlChart("PnL")
        self.win_rate = PnlChart("Win Rate History")
        self.drawdown = PnlChart("Drawdown")
        grid.addWidget(self.equity, 0, 0)
        grid.addWidget(self.pnl, 0, 1)
        grid.addWidget(self.win_rate, 1, 0)
        grid.addWidget(self.drawdown, 1, 1)
        root.addLayout(grid)

    def update_state(self, state: dict) -> None:
        analytics = state.get("analytics", {}) if isinstance(state.get("analytics"), dict) else {}
        self.equity.plot(_to_float_list(analytics.get("equity_curve", [])))
        self.pnl.plot(_to_float_list(analytics.get("pnl_curve", [])))
        self.win_rate.plot(_to_float_list(analytics.get("win_rate_history", [])))
        self.drawdown.plot(_to_float_list(analytics.get("drawdown_curve", [])))


def _to_float_list(values: list) -> list[float]:
    out: list[float] = []
    for v in values:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out
