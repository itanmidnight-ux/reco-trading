from __future__ import annotations

from pathlib import Path

import yaml
from PySide6.QtWidgets import QLabel, QTabWidget, QTextEdit, QVBoxLayout, QWidget


class SettingsTab(QWidget):
    """Read-only configuration inspector."""

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        title = QLabel("Settings (Read-Only)")
        title.setObjectName("sectionTitle")
        root.addWidget(title)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)
        for cfg in ["trading.yaml", "risk.yaml", "strategy.yaml"]:
            self.tabs.addTab(self._make_view(cfg), cfg)

    def _make_view(self, name: str) -> QWidget:
        editor = QTextEdit()
        editor.setReadOnly(True)
        path = Path("config") / name
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            text = yaml.dump(data, sort_keys=False)
        except Exception:
            text = "--"
        editor.setPlainText(text)
        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.addWidget(editor)
        return wrap

    def update_state(self, state: dict) -> None:
        _ = state
