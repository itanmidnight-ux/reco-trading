from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout


class StatCard(QFrame):
    def __init__(self, label: str, value: str = "-", compact: bool = False) -> None:
        super().__init__()
        self.setObjectName("metricCard")
        self._base_value_object_name = "smallMetricValue" if compact else "metricValue"
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        self.label = QLabel(label)
        self.label.setObjectName("metricLabel")
        self.value = QLabel(value)
        self.value.setObjectName(self._base_value_object_name)
        self.badge = QLabel("")
        self.badge.setVisible(False)
        self.badge.setObjectName("metricBadge")
        layout.addWidget(self.label)
        layout.addWidget(self.value)
        layout.addWidget(self.badge)
        self._anim = QPropertyAnimation(self.value, b"windowOpacity", self)
        self._anim.setDuration(260)
        self._anim.setStartValue(0.55)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.set_tone("neutral")

    def set_value(self, value: str, tone: str | None = None, badge: str | None = None) -> None:
        if self.value.text() == value:
            if tone:
                self.set_tone(tone)
            if badge is not None:
                self.set_badge(badge)
            return
        self.value.setText(value)
        self._anim.stop()
        self._anim.start()
        if tone:
            self.set_tone(tone)
        if badge is not None:
            self.set_badge(badge)

    def set_tone(self, tone: str) -> None:
        palette = {
            "positive": ("#173a33", "#1f8f73", "#b6ffea"),
            "negative": ("#3d1b27", "#b8425d", "#ffd5df"),
            "warning": ("#3e3115", "#b88a1f", "#ffedbb"),
            "info": ("#152f4b", "#2c6fd6", "#d7e6ff"),
            "neutral": ("#1b2640", "#314262", "#edf2ff"),
        }
        background, border, text = palette.get(tone, palette["neutral"])
        self.setStyleSheet(
            "QFrame#metricCard {"
            f"background: {background}; border: 1px solid {border}; border-radius: 14px;"
            "}"
        )
        self.value.setStyleSheet(f"color: {text};")
        self.label.setStyleSheet("letter-spacing: 0.6px;")

    def set_badge(self, text: str) -> None:
        content = text.strip()
        self.badge.setVisible(bool(content))
        if not content:
            self.badge.clear()
            return
        self.badge.setText(content.upper())
        self.badge.setStyleSheet(
            "background: rgba(255,255,255,0.08);"
            "border: 1px solid rgba(255,255,255,0.12);"
            "border-radius: 8px;"
            "padding: 3px 8px;"
            "color: #dbe7ff;"
            "font-size: 10px;"
            "font-weight: 700;"
        )
