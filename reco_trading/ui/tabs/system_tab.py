from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout, QWidget

from reco_trading.ui.widgets.stat_card import StatCard


class SystemTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        title = QLabel("System")
        title.setObjectName("sectionTitle")
        root.addWidget(title)

        panel = QFrame()
        panel.setObjectName("panelCard")
        root.addWidget(panel, 1)
        layout = QGridLayout(panel)

        self.cards = {
            "cpu": StatCard("CPU Usage", compact=True),
            "memory": StatCard("Memory Usage", compact=True),
            "uptime": StatCard("Uptime", compact=True),
            "loop": StatCard("Loop Frequency", compact=True),
            "latency": StatCard("Exchange Latency", compact=True),
        }
        for i, card in enumerate(self.cards.values()):
            layout.addWidget(card, i // 2, i % 2)
        layout.setRowStretch(3, 1)

    def update_state(self, state: dict[str, Any]) -> None:
        system = state.get("system", {}) or {}
        self.cards["cpu"].set_value(_safe(system.get("cpu_usage_pct", state.get("cpu_usage_pct")), "%"))
        self.cards["memory"].set_value(_safe(system.get("memory_usage_mb", state.get("memory_usage_mb")), " MB"))
        self.cards["uptime"].set_value(_safe(system.get("uptime", system.get("uptime_seconds", "-"))))
        self.cards["loop"].set_value(_safe(system.get("loop_frequency_hz", state.get("loop_frequency_hz")), " Hz"))
        self.cards["latency"].set_value(_safe(system.get("api_latency_ms", state.get("api_latency_ms")), " ms"))


def _safe(value: Any, suffix: str = "") -> str:
    if value in (None, ""):
        return "-"
    return f"{value}{suffix}"
