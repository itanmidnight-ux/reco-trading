from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

from reco_trading.ui.widgets.stat_card import StatCard


class RiskTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        title = QLabel("Risk Center")
        title.setObjectName("sectionTitle")
        root.addWidget(title)

        subtitle = QLabel("Exposure and drawdown controls")
        subtitle.setObjectName("metricLabel")
        root.addWidget(subtitle)

        panel = QFrame()
        panel.setObjectName("panelCard")
        root.addWidget(panel)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 12, 12, 12)

        layout = QGridLayout()
        self.cards = {}
        keys = ["risk_per_trade", "max_concurrent_trades", "daily_drawdown", "current_exposure"]
        for i, key in enumerate(keys):
            card = StatCard(key.replace("_", " ").title(), compact=True)
            self.cards[key] = card
            layout.addWidget(card, i // 2, i % 2)
        panel_layout.addLayout(layout)

        self.exposure_bar = QProgressBar()
        self.exposure_bar.setRange(0, 100)
        panel_layout.addWidget(self.exposure_bar)
        self._anim = QPropertyAnimation(self.exposure_bar, b"value", self)
        self._anim.setDuration(300)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def update_state(self, state: dict) -> None:
        metrics = state.get("risk_metrics", {})
        for key, card in self.cards.items():
            card.set_value(str(metrics.get(key, "-")))
        try:
            exposure = int(float(metrics.get("current_exposure", 0)) * 100)
        except (TypeError, ValueError):
            exposure = 0
        self._anim.stop()
        self._anim.setStartValue(self.exposure_bar.value())
        self._anim.setEndValue(max(0, min(100, exposure)))
        self._anim.start()
