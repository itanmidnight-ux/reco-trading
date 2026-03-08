from __future__ import annotations

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QTextEdit, QVBoxLayout, QWidget


class LogsTab(QWidget):
    COLORS = {"INFO": "white", "WARNING": "yellow", "ERROR": "red"}

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        layout.addWidget(self.text)

    def add_log(self, entry: dict) -> None:
        color = self.COLORS.get(entry.get("level", "INFO"), "white")
        self.text.setTextColor(QColor(color))
        self.text.append(f"[{entry.get('level', 'INFO')}] {entry.get('time', '')} {entry.get('message', '')}")
        self.text.moveCursor(self.text.textCursor().End)
