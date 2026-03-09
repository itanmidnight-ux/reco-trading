from __future__ import annotations

from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QLabel, QPushButton, QTextEdit, QHBoxLayout, QVBoxLayout, QWidget


class LogsTab(QWidget):
    COLORS = {"INFO": "#e6e8ee", "WARNING": "#f0b90b", "ERROR": "#ea3943"}

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        head = QHBoxLayout()
        title = QLabel("System Logs")
        title.setObjectName("sectionTitle")
        clear = QPushButton("Clear")
        clear.clicked.connect(lambda: self.text.clear())
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(clear)
        layout.addLayout(head)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(QFont("Consolas", 10))
        layout.addWidget(self.text)

    def add_log(self, entry: dict) -> None:
        level = entry.get("level", "INFO").upper()
        self.text.setTextColor(QColor(self.COLORS.get(level, "#e6e8ee")))
        self.text.append(f"[{entry.get('time', '')}] [{level}] {entry.get('message', '')}")
        self.text.moveCursor(self.text.textCursor().End)
