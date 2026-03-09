from __future__ import annotations

import platform
from typing import Any

from PySide6.QtWidgets import QGridLayout, QWidget

from reco_trading.ui.widgets.stat_card import StatCard


class SystemTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QGridLayout(self)
        self.cards = {
            "version": StatCard("Bot Version", compact=True),
            "python": StatCard("Python Version", compact=True),
            "api": StatCard("API Connectivity", compact=True),
            "database": StatCard("Database Status", compact=True),
            "latency": StatCard("Latency", compact=True),
        }
        for i, card in enumerate(self.cards.values()):
            layout.addWidget(card, i // 3, i % 3)
        self.cards["python"].set_value(platform.python_version())

    def update_state(self, state: dict[str, Any]) -> None:
        system = state.get("system", {})
        self.cards["version"].set_value(str(state.get("bot_version", "-")))
        self.cards["api"].set_value(str(system.get("exchange_status", "UNKNOWN")))
        self.cards["database"].set_value(str(system.get("database_status", "UNKNOWN")))
        self.cards["latency"].set_value(f"{system.get('api_latency_ms', '-') } ms")
