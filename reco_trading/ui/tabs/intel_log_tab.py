from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QTextCursor
from PySide6.QtWidgets import QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget


class IntelLogTab(QWidget):
    COLORS = {
        "EXIT_INTELLIGENCE_HOLD": "#8f9bb3",
        "EXIT_INTELLIGENCE_GIVEBACK": "#f0b90b",
        "EXIT_INTELLIGENCE_MOMENTUM_FADE": "#f0b90b",
        "EXIT_INTELLIGENCE_STRUCTURE_WEAKNESS": "#f0b90b",
        "EXIT_INTELLIGENCE_TIME_EFFICIENCY": "#f0b90b",
        "EXIT_INTELLIGENCE_COST_PRESSURE": "#f0b90b",
        "AUTO_IMPROVE": "#00ff00",
        "OPTIMIZATION": "#00ffff",
        "DEFENSIVE": "#ff6600",
        "AGGRESSIVE": "#ff00ff",
        "MARKET_ADJUST": "#ffff00",
    }

    def __init__(self) -> None:
        super().__init__()
        self._rendered_signature: tuple[tuple[str, str], ...] = tuple()
        self._log_filter = "all"
        layout = QVBoxLayout(self)
        
        head = QHBoxLayout()
        title = QLabel("Intelligence Log")
        title.setObjectName("sectionTitle")
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_logs)
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["all", "exit_intelligence", "auto_improver", "optimizer"])
        self.filter_combo.currentTextChanged.connect(self._on_filter_changed)
        
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(QLabel("Filter:"))
        head.addWidget(self.filter_combo)
        head.addWidget(self.clear_btn)
        layout.addLayout(head)

        subtitle = QLabel("Exit intelligence and auto-improvement audit trail")
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

    def _on_filter_changed(self, text: str) -> None:
        self._log_filter = text
        self._rendered_signature = tuple()

    def _clear_logs(self) -> None:
        self.text.clear()
        self._rendered_signature = tuple()
        self.summary.setText("Entries: 0")

    def update_state(self, state: dict) -> None:
        all_logs = []
        
        exit_intel = state.get("exit_intelligence", {}) if isinstance(state, dict) else {}
        exit_logs = exit_intel.get("logs", []) if isinstance(exit_intel, dict) else []
        for entry in exit_logs:
            if isinstance(entry, dict):
                entry_copy = dict(entry)
                entry_copy["log_type"] = "exit_intelligence"
                all_logs.append(entry_copy)
        
        auto_improve = state.get("auto_improve_logs", []) if isinstance(state, dict) else []
        for entry in auto_improve:
            if isinstance(entry, dict):
                entry_copy = dict(entry)
                entry_copy["log_type"] = "auto_improver"
                all_logs.append(entry_copy)
        
        optimizer = state.get("hyperopt", {}) if isinstance(state, dict) else {}
        if optimizer.get("status") == "running":
            all_logs.append({
                "time": optimizer.get("last_update", ""),
                "log_type": "optimizer",
                "message": f"Optimization running - Trial {optimizer.get('current_trial', 0)}/{optimizer.get('total_trials', 0)}",
                "score": optimizer.get("best_score", 0),
            })
        
        all_logs.sort(key=lambda x: x.get("time", ""), reverse=True)
        
        filtered_logs = all_logs
        if self._log_filter != "all":
            filtered_logs = [l for l in all_logs if l.get("log_type") == self._log_filter]
        
        signature = tuple(
            (
                str(entry.get("time", "")),
                f"{entry.get('log_type', '')}-{entry.get('message', '')}",
            )
            for entry in filtered_logs[-300:]
        )
        if signature == self._rendered_signature:
            return

        self.text.clear()
        self._rendered_signature = tuple()
        if not filtered_logs:
            self.summary.setText("Entries: 0")
            self.text.setTextColor(QColor("#888888"))
            self.text.append("No logs available yet. The system will log:")
            self.text.append("  - Exit intelligence decisions")
            self.text.append("  - Auto-improvement system decisions")
            self.text.append("  - Optimization status")
            self.text.append("  - Market regime changes")
            self.text.append("  - Filter adjustments")
            return

        rendered = 0
        for entry in filtered_logs[-300:]:
            if not isinstance(entry, dict):
                continue
            
            log_type = entry.get("log_type", "unknown")
            
            if log_type == "exit_intelligence":
                reason = str(entry.get("reason", "EXIT_INTELLIGENCE_HOLD"))
                self.text.setTextColor(QColor(self.COLORS.get(reason, "#e6e8ee")))
                self.text.append(
                    f"[{entry.get('time', '--:--:--')}] [EXIT] "
                    f"trade={entry.get('trade_id', '-')} bars={entry.get('bars_held', '-')} "
                    f"score={float(entry.get('score', 0.0)):.4f} "
                    f"threshold={float(entry.get('threshold', 0.0)):.4f} "
                    f"reason={reason}"
                )
            elif log_type == "auto_improver":
                message = entry.get("message", "")
                self.text.setTextColor(QColor(self.COLORS.get("AUTO_IMPROVE", "#00ff00")))
                self.text.append(
                    f"[{entry.get('time', '--:--:--')}] [AUTO-IMPROVE] {message}"
                )
            elif log_type == "optimizer":
                message = entry.get("message", "")
                self.text.setTextColor(QColor(self.COLORS.get("OPTIMIZATION", "#00ffff")))
                self.text.append(
                    f"[{entry.get('time', '--:--:--')}] [OPTIMIZER] {message}"
                )
            
            rendered += 1
            self._rendered_signature = (
                *self._rendered_signature,
                (
                    str(entry.get("time", "")),
                    f"{entry.get('log_type', '')}-{entry.get('message', '')}",
                ),
            )[-300:]

        cursor = self.text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.text.setTextCursor(cursor)
        self.summary.setText(f"Entries: {rendered}")
