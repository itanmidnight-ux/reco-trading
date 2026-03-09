from __future__ import annotations

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget


class LogsTab(QWidget):
    COLORS = {"INFO": "#e6e8ee", "WARNING": "#f0b90b", "ERROR": "#ea3943"}

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)

        head = QHBoxLayout()
        title = QLabel("Logs")
        title.setObjectName("sectionTitle")
        self.summary = QLabel("INFO: 0 | WARNING: 0 | ERROR: 0")
        self.summary.setObjectName("metricLabel")
        clear = QPushButton("Clear")
        clear.clicked.connect(self._clear)
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(self.summary)
        head.addWidget(clear)
        layout.addLayout(head)

        panel = QFrame()
        panel.setObjectName("panelCard")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(10, 10, 10, 10)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Timestamp", "Level", "Message"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(False)
        panel_layout.addWidget(self.table)
        layout.addWidget(panel, 1)

        self._counts = {"INFO": 0, "WARNING": 0, "ERROR": 0}

    def _clear(self) -> None:
        self.table.setRowCount(0)
        self._counts = {"INFO": 0, "WARNING": 0, "ERROR": 0}
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        self.summary.setText(
            f"INFO: {self._counts.get('INFO', 0)} | WARNING: {self._counts.get('WARNING', 0)} | ERROR: {self._counts.get('ERROR', 0)}"
        )

    def add_log(self, entry: dict) -> None:
        try:
            row = self.table.rowCount()
            self.table.insertRow(row)
            timestamp = str(entry.get("time", entry.get("timestamp", "-")))
            level = str(entry.get("level", "INFO")).upper()
            message = str(entry.get("message", "-"))
            for col, value in enumerate((timestamp, level, message)):
                item = QTableWidgetItem(value)
                if col == 1:
                    item.setForeground(QColor(self.COLORS.get(level, "#e6e8ee")))
                self.table.setItem(row, col, item)
            self._counts[level] = self._counts.get(level, 0) + 1
            self._refresh_summary()
            self.table.scrollToBottom()
        except Exception:
            return

    def update_state(self, state: dict) -> None:
        logs = state.get("logs", []) or []
        self._clear()
        for entry in logs[-500:]:
            self.add_log(entry or {})
