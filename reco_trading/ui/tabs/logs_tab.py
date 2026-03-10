from __future__ import annotations

from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget


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

        subtitle = QLabel("Realtime event stream and diagnostics")
        subtitle.setObjectName("metricLabel")
        layout.addWidget(subtitle)

        self.summary = QLabel("INFO: 0 | WARNING: 0 | ERROR: 0")
        self.summary.setObjectName("smallMetricValue")
        layout.addWidget(self.summary)

        panel = QFrame()
        panel.setObjectName("panelCard")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(10, 10, 10, 10)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(QFont("Consolas", 10))
        panel_layout.addWidget(self.text)
        layout.addWidget(panel)
        self._counts = {"INFO": 0, "WARNING": 0, "ERROR": 0}

    def add_log(self, entry: dict) -> None:
        level = entry.get("level", "INFO").upper()
        self.text.setTextColor(QColor(self.COLORS.get(level, "#e6e8ee")))
        self.text.append(f"[{entry.get('time', '')}] [{level}] {entry.get('message', '')}")
        self.text.moveCursor(self.text.textCursor().End)
        self._counts[level] = self._counts.get(level, 0) + 1
        self.summary.setText(
            f"INFO: {self._counts.get('INFO', 0)} | WARNING: {self._counts.get('WARNING', 0)} | ERROR: {self._counts.get('ERROR', 0)}"
        )

    def update_state(self, state: dict) -> None:
        logs = state.get("logs", [])
        if not logs:
            return
        self.text.clear()
        self._counts = {"INFO": 0, "WARNING": 0, "ERROR": 0}
        for entry in logs[-300:]:
            self.add_log(entry)
