from __future__ import annotations

from PySide6.QtCore import QEasingCurve, Property, QPropertyAnimation
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout


class StatCard(QFrame):
    def __init__(self, label: str, value: str = "-") -> None:
        super().__init__()
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        self.label = QLabel(label)
        self.label.setObjectName("cardTitle")
        self.value = QLabel(value)
        self.value.setObjectName("cardValue")
        layout.addWidget(self.label)
        layout.addWidget(self.value)
        self._opacity = 1.0
        self._anim = QPropertyAnimation(self, b"opacity")
        self._anim.setDuration(160)
        self._anim.setEasingCurve(QEasingCurve.InOutQuad)

    def set_value(self, value: str) -> None:
        if self.value.text() != value:
            self._anim.stop()
            self._anim.setStartValue(0.5)
            self._anim.setEndValue(1.0)
            self._anim.start()
        self.value.setText(value)

    def get_opacity(self) -> float:
        return self._opacity

    def set_opacity(self, value: float) -> None:
        self._opacity = value
        self.value.setStyleSheet(f"opacity: {value};")

    opacity = Property(float, get_opacity, set_opacity)
