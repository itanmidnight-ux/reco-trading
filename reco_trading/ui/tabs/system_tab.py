from __future__ import annotations

import time

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
            "cpu": StatCard("CPU usage"),
            "ram": StatCard("RAM usage"),
            "uptime": StatCard("Bot uptime"),
            "db": StatCard("Database status"),
            "latency": StatCard("API latency"),
            "sync": StatCard("Last server sync time"),
        }
        for i, card in enumerate(self.cards.values()):
            layout.addWidget(card, i // 3, i % 3)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_system)
        self.timer.start(2000)

    def update_state(self, state: dict) -> None:
        system = state.get("system", {})
        self.cards["db"].set_value(str(system.get("database_status", "UNKNOWN")))
        self.cards["latency"].set_value(f"{system.get('api_latency_ms', 0.0):.2f} ms")
        self.cards["sync"].set_value(str(system.get("last_server_sync", "-")))

    def update_system(self) -> None:
        self.cards["cpu"].set_value(f"{psutil.cpu_percent(interval=None):.1f}%")
        self.cards["ram"].set_value(f"{psutil.virtual_memory().percent:.1f}%")
        self.cards["uptime"].set_value(f"{int(time.time() - self.start)} s")
