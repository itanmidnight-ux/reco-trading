from __future__ import annotations

from PySide6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

from reco_trading.ui.components.formatting import fmt_pct


class RiskTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        title = QLabel("Risk")
        title.setObjectName("sectionTitle")
        root.addWidget(title)

        self.risk_per_trade = QLabel("Risk per trade: --")
        self.exposure = QLabel("Current exposure: --")
        self.max_drawdown = QLabel("Max drawdown: --")
        self.daily_limit = QLabel("Daily loss limit: --")
        for w in [self.risk_per_trade, self.exposure, self.max_drawdown, self.daily_limit]:
            root.addWidget(w)

        self.exposure_gauge = QProgressBar()
        self.exposure_gauge.setRange(0, 100)
        self.drawdown_gauge = QProgressBar()
        self.drawdown_gauge.setRange(0, 100)
        root.addWidget(self.exposure_gauge)
        root.addWidget(self.drawdown_gauge)

    def update_state(self, state: dict) -> None:
        metrics = state.get("risk_metrics", {}) if isinstance(state.get("risk_metrics"), dict) else {}
        self.risk_per_trade.setText(f"Risk per trade: {metrics.get('risk_per_trade', '--')}")
        exp = _to_pct(metrics.get("current_exposure", 0))
        self.exposure.setText(f"Current exposure: {exp:.1f}%")
        dd = _to_pct(abs(metrics.get("max_drawdown", metrics.get("daily_drawdown", 0))))
        self.max_drawdown.setText(f"Max drawdown: {dd:.1f}%")
        self.daily_limit.setText(f"Daily loss limit: {metrics.get('daily_loss_limit', '--')}")
        self.exposure_gauge.setValue(int(max(0, min(100, exp))))
        self.drawdown_gauge.setValue(int(max(0, min(100, dd))))


def _to_pct(value) -> float:
    try:
        val = float(str(value).replace("%", ""))
    except (TypeError, ValueError):
        return 0.0
    return val * 100 if abs(val) <= 1 else val
