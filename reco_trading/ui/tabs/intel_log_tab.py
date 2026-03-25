from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QTextCursor
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget


class IntelLogTab(QWidget):
    COLORS = {
        "EXIT_INTELLIGENCE_HOLD": "#8f9bb3",
        "EXIT_INTELLIGENCE_GIVEBACK": "#f0b90b",
        "EXIT_INTELLIGENCE_MOMENTUM_FADE": "#f0b90b",
        "EXIT_INTELLIGENCE_STRUCTURE_WEAKNESS": "#f0b90b",
        "EXIT_INTELLIGENCE_TIME_EFFICIENCY": "#f0b90b",
        "EXIT_INTELLIGENCE_COST_PRESSURE": "#f0b90b",
    }

    def __init__(self) -> None:
        super().__init__()
        self._rendered_signature: tuple[tuple[str, str], ...] = tuple()
        layout = QVBoxLayout(self)
        head = QHBoxLayout()
        title = QLabel("Intelligence Log")
        title.setObjectName("sectionTitle")
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_logs)
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(self.clear_btn)
        layout.addLayout(head)

        subtitle = QLabel("Exit intelligence audit trail (score/threshold/reason)")
        subtitle.setObjectName("metricLabel")
        layout.addWidget(subtitle)

        self.summary = QLabel("Entries: 0")
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

    def _clear_logs(self) -> None:
        self.text.clear()
        self._rendered_signature = tuple()
        self.summary.setText("Entries: 0")

    def update_state(self, state: dict) -> None:
        intel_payload = state.get("exit_intelligence", {}) if isinstance(state, dict) else {}
        logs = intel_payload.get("logs", []) if isinstance(intel_payload, dict) else []
        signature = tuple(
            (
                str(entry.get("time", "")),
                f"{entry.get('trade_id', '-')}-{entry.get('score', 0)}-{entry.get('threshold', 0)}-{entry.get('reason', '')}",
            )
            for entry in logs[-300:]
            if isinstance(entry, dict)
        )
        if signature == self._rendered_signature:
            return

        self.text.clear()
        self._rendered_signature = tuple()
        if not logs:
            self.summary.setText("Entries: 0")
            return

        rendered = 0
        for entry in logs[-300:]:
            if not isinstance(entry, dict):
                continue
            reason = str(entry.get("reason", "EXIT_INTELLIGENCE_HOLD"))
            self.text.setTextColor(QColor(self.COLORS.get(reason, "#e6e8ee")))
            self.text.append(
                f"[{entry.get('time', '--:--:--')}] "
                f"trade={entry.get('trade_id', '-')} bars={entry.get('bars_held', '-')} "
                f"score={float(entry.get('score', 0.0)):.4f} "
                f"threshold={float(entry.get('threshold', 0.0)):.4f} "
                f"reason={reason} "
                f"codes={','.join(entry.get('codes', []) or []) or 'NONE'}"
            )
            rendered += 1
            self._rendered_signature = (
                *self._rendered_signature,
                (
                    str(entry.get("time", "")),
                    f"{entry.get('trade_id', '-')}-{entry.get('score', 0)}-{entry.get('threshold', 0)}-{entry.get('reason', '')}",
                ),
            )[-300:]

        cursor = self.text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.text.setTextCursor(cursor)
        self.summary.setText(f"Entries: {rendered}")
