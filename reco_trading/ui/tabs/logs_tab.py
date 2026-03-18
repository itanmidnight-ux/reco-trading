from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QTextCursor
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget

from reco_trading.ui.state_manager import StateManager


class LogsTab(QWidget):
    COLORS = {"INFO": "#e6e8ee", "WARNING": "#f0b90b", "ERROR": "#ea3943"}

    def __init__(self, state_manager: StateManager | None = None) -> None:
        super().__init__()
        self.state_manager = state_manager
        self._rendered_signature: tuple[tuple[str, str, str], ...] = tuple()
        layout = QVBoxLayout(self)
        head = QHBoxLayout()
        title = QLabel("System Logs")
        title.setObjectName("sectionTitle")
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_logs)
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(self.clear_btn)
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

    def _clear_logs(self) -> None:
        self.text.clear()
        self._counts = {"INFO": 0, "WARNING": 0, "ERROR": 0}
        self._rendered_signature = tuple()
        self.summary.setText("INFO: 0 | WARNING: 0 | ERROR: 0")
        if self.state_manager:
            self.state_manager.clear_logs()

    def add_log(self, entry: dict) -> None:
        level = entry.get("level", "INFO").upper()
        self.text.setTextColor(QColor(self.COLORS.get(level, "#e6e8ee")))
        self.text.append(f"[{entry.get('time', '')}] [{level}] {entry.get('message', '')}")
        cursor = self.text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.text.setTextCursor(cursor)
        self._counts[level] = self._counts.get(level, 0) + 1
        self._rendered_signature = (
            *self._rendered_signature,
            (str(entry.get("time", "")), str(level), str(entry.get("message", ""))),
        )[-300:]
        self.summary.setText(
            f"INFO: {self._counts.get('INFO', 0)} | WARNING: {self._counts.get('WARNING', 0)} | ERROR: {self._counts.get('ERROR', 0)}"
        )

    def update_state(self, state: dict) -> None:
        logs = state.get("logs", [])
        signature = tuple(
            (str(entry.get("time", "")), str(entry.get("level", "INFO")).upper(), str(entry.get("message", "")))
            for entry in logs[-300:]
        )
        if signature == self._rendered_signature:
            return
        if not logs:
            self.text.clear()
            self._counts = {"INFO": 0, "WARNING": 0, "ERROR": 0}
            self._rendered_signature = tuple()
            self.summary.setText("INFO: 0 | WARNING: 0 | ERROR: 0")
            return
        self.text.clear()
        self._counts = {"INFO": 0, "WARNING": 0, "ERROR": 0}
        self._rendered_signature = tuple()
        for entry in logs[-300:]:
            self.add_log(entry)
