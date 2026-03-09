from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout


class StatCard(QFrame):
    def __init__(self, label: str, value: str = "-", compact: bool = False) -> None:
        super().__init__()
        self.setObjectName("metricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        self.label = QLabel(label)
        self.label.setObjectName("metricLabel")
        self.value = QLabel(value)
        self.value.setObjectName("smallMetricValue" if compact else "metricValue")
        layout.addWidget(self.label)
        layout.addWidget(self.value)
        self._anim = QPropertyAnimation(self.value, b"windowOpacity", self)
        self._anim.setDuration(260)
        self._anim.setStartValue(0.55)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def set_value(self, value: str) -> None:
        if self.value.text() == value:
            return
        self.value.setText(value)
        self._anim.stop()
        self._anim.start()
