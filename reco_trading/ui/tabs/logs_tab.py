from __future__ import annotations

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QTextEdit, QVBoxLayout, QWidget


class LogsTab(QWidget):
    COLORS = {"INFO": "#cbd5e1", "WARNING": "#f59e0b", "ERROR": "#ef4444"}

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        title = QLabel("Logs")
        title.setObjectName("sectionTitle")
        root.addWidget(title)

        row = QHBoxLayout()
        self.filter = QComboBox()
        self.filter.addItems(["ALL", "INFO", "WARNING", "ERROR"])
        self.filter.currentTextChanged.connect(self._render)
        row.addWidget(QLabel("Severity Filter"))
        row.addWidget(self.filter)
        row.addStretch(1)
        root.addLayout(row)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        root.addWidget(self.text)
        self._entries: list[dict] = []

    def update_state(self, state: dict) -> None:
        self._entries = list(state.get("logs", []))[-500:]
        self._render()

    def _render(self) -> None:
        level = self.filter.currentText()
        self.text.clear()
        for e in self._entries:
            sev = str(e.get("level", "INFO")).upper()
            if level != "ALL" and sev != level:
                continue
            self.text.setTextColor(QColor(self.COLORS.get(sev, "#cbd5e1")))
            self.text.append(f"[{e.get('time', '--')}] [{sev}] {e.get('message', '')}")
        self.text.moveCursor(self.text.textCursor().End)
