from __future__ import annotations

import time
from typing import Any

import psutil
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QGridLayout, QWidget

from reco_trading.ui.widgets.stat_card import StatCard


class SystemTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QGridLayout(self)
        self.start = time.time()
        self.cards = {
            "binance": StatCard("Binance status"),
            "db": StatCard("PostgreSQL status"),
            "redis": StatCard("Redis status"),
            "latency": StatCard("API latency"),
            "uptime": StatCard("Bot uptime"),
            "ram": StatCard("Memory usage"),
        }
        for i, card in enumerate(self.cards.values()):
            layout.addWidget(card, i // 3, i % 3)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_system)
        self.timer.start(2000)

    def update_state(self, state: dict[str, Any]) -> None:
        system = state.get("system") or {}
        self.cards["db"].set_value(str(system.get("database_status", "UNKNOWN")))
        self.cards["binance"].set_value(str(system.get("exchange_status", "UNKNOWN")))
        self.cards["redis"].set_value(str(system.get("redis_status", "UNKNOWN")))
        self.cards["latency"].set_value(f"{_fmt_num(system.get('api_latency_ms'), 2)} ms")

    def update_system(self) -> None:
        self.cards["ram"].set_value(f"{psutil.virtual_memory().percent:.1f}%")
        self.cards["uptime"].set_value(f"{int(time.time() - self.start)} s")


def _fmt_num(value: Any, digits: int) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"
