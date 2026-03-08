from __future__ import annotations

from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout


class StatCard(QFrame):
    def __init__(self, label: str, value: str = "-") -> None:
        super().__init__()
        self.setObjectName("statCard")
        layout = QVBoxLayout(self)
        self.label = QLabel(label)
        self.value = QLabel(value)
        self.value.setStyleSheet("font-size: 18px; font-weight: 700;")
        layout.addWidget(self.label)
        layout.addWidget(self.value)
        self.setStyleSheet("#statCard { border: 1px solid #444; border-radius: 8px; padding: 8px; }")

    def set_value(self, value: str) -> None:
        self.value.setText(value)
