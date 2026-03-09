from __future__ import annotations

import platform
import time
from typing import Any

import psutil
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QGridLayout, QWidget

from reco_trading.ui.widgets.stat_card import StatCard


class SystemTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.start = time.time()
        layout = QGridLayout(self)
        self.cards = {
            "version": StatCard("Bot Version"),
            "python": StatCard("Python Version", platform.python_version()),
            "api": StatCard("API Connectivity"),
            "db": StatCard("Database Status"),
            "latency": StatCard("Latency"),
            "ram": StatCard("Memory Usage"),
            "uptime": StatCard("Uptime"),
        }
        self.cards["version"].set_value("reco-trading")
        for i, card in enumerate(self.cards.values()):
            layout.addWidget(card, i // 3, i % 3)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_system)
        self.timer.start(1000)

    def update_state(self, state: dict[str, Any]) -> None:
        system = state.get("system") or {}
        self.cards["db"].set_value(str(system.get("database_status", "UNKNOWN")))
        self.cards["api"].set_value(str(system.get("exchange_status", "UNKNOWN")))
        self.cards["latency"].set_value(f"{_fmt_num(system.get('api_latency_ms'), 2)} ms")

    def update_system(self) -> None:
        self.cards["ram"].set_value(f"{psutil.virtual_memory().percent:.1f}%")
        self.cards["uptime"].set_value(f"{int(time.time() - self.start)} s")


def _fmt_num(value: Any, digits: int) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"
