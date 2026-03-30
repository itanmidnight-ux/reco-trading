from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QLabel


class FlashValueLabel(QLabel):
    """Label that briefly flashes a color when value changes."""

    def __init__(self, text: str = "--") -> None:
        super().__init__(text)
        self._reset_timer = QTimer(self)
        self._reset_timer.setSingleShot(True)
        self._reset_timer.timeout.connect(self._reset_style)
        self.setObjectName("valueLabel")

    def set_value(self, text: str, positive: bool | None = None) -> None:
        changed = text != self.text()
        self.setText(text)
        if changed:
            if positive is True:
                self._flash(QColor("#22c55e"))
            elif positive is False:
                self._flash(QColor("#ef4444"))
            else:
                self._flash(QColor("#38bdf8"))

    def _flash(self, color: QColor) -> None:
        self.setStyleSheet(f"color: {color.name()}; font-weight: 700;")
        self._reset_timer.start(420)

    def _reset_style(self) -> None:
        self.setStyleSheet("")


class HoverPulseMixin:
    def setup_hover_anim(self) -> None:
        self._hover_anim = QPropertyAnimation(self, b"windowOpacity", self)
        self._hover_anim.setDuration(160)
        self._hover_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def enterEvent(self, event) -> None:  # type: ignore[override]
        self._hover_anim.stop()
        self._hover_anim.setStartValue(self.windowOpacity())
        self._hover_anim.setEndValue(0.96)
        self._hover_anim.start()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        self._hover_anim.stop()
        self._hover_anim.setStartValue(self.windowOpacity())
        self._hover_anim.setEndValue(1.0)
        self._hover_anim.start()
        super().leaveEvent(event)
