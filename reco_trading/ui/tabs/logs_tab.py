from __future__ import annotations

from PySide6.QtWidgets import QTextEdit, QVBoxLayout, QWidget


class LogsTab(QWidget):
    COLORS = {"INFO": "#e6e8ee", "WARNING": "#f0b90b", "ERROR": "#ea3943"}

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setStyleSheet("font-family: 'JetBrains Mono', 'Courier New', monospace;")
        layout.addWidget(self.text)

    def add_log(self, entry: dict) -> None:
        level = str(entry.get("level", "INFO")).upper()
        color = self.COLORS.get(level, "#e6e8ee")
        line = f"<span style='color:{color}'>[{level}] {entry.get('time', '')} {entry.get('message', '')}</span>"
        self.text.append(line)
