from __future__ import annotations

import platform
from typing import Any

from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QListWidget, QVBoxLayout, QWidget

from reco_trading.ui.widgets.stat_card import StatCard


class SystemTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        title = QLabel("System Health")
        title.setObjectName("sectionTitle")
        root.addWidget(title)

        subtitle = QLabel("Runtime, connectivity, infra and telemetry diagnostics")
        subtitle.setObjectName("metricLabel")
        root.addWidget(subtitle)

        panel = QFrame()
        panel.setObjectName("panelCard")
        root.addWidget(panel)
        panel_layout = QGridLayout(panel)
        panel_layout.setContentsMargins(12, 12, 12, 12)

        self.cards = {
            "version": StatCard("Bot Version", compact=True),
            "python": StatCard("Python Version", compact=True),
            "api": StatCard("API Connectivity", compact=True),
            "database": StatCard("Database Status", compact=True),
            "latency": StatCard("Latency", compact=True),
            "memory": StatCard("Memory Usage", compact=True),
            "redis": StatCard("Redis", compact=True),
        }
        self.cards["version"].set_value("reco-trading")
        for i, card in enumerate(self.cards.values()):
            panel_layout.addWidget(card, i // 3, i % 3)
        self.cards["python"].set_value(platform.python_version())

        self.health_badge = QLabel("Health: UNKNOWN")
        self.health_badge.setObjectName("smallMetricValue")
        panel_layout.addWidget(self.health_badge, 3, 0, 1, 3)

        self.events = QListWidget()
        self.events.addItems(["System diagnostics pending..."])
        panel_layout.addWidget(self.events, 4, 0, 1, 3)

    def update_state(self, state: dict[str, Any]) -> None:
        system = state.get("system", {})
        exchange_status = str(system.get("exchange_status", "UNKNOWN"))
        database_status = str(system.get("database_status", "UNKNOWN"))
        redis_status = str(system.get("redis_status", "UNKNOWN"))

        self.cards["version"].set_value(str(state.get("bot_version") or "reco-trading"))
        self.cards["api"].set_value(exchange_status)
        self.cards["database"].set_value(database_status)
        self.cards["latency"].set_value(f"{system.get('api_latency_ms', '-') } ms")
        self.cards["memory"].set_value(f"{system.get('memory_usage_mb', '-') } MB")
        self.cards["redis"].set_value(redis_status)

        if all(value in {"OK", "CONNECTED", "ONLINE", "UNKNOWN"} for value in (exchange_status, database_status, redis_status)):
            self.health_badge.setText("Health: STABLE")
            self.health_badge.setStyleSheet("color:#16c784;")
        else:
            self.health_badge.setText("Health: DEGRADED")
            self.health_badge.setStyleSheet("color:#f0b90b;")

        self.events.clear()
        self.events.addItems(
            [
                f"Uptime: {system.get('uptime_seconds', 0)}s",
                f"Last server sync: {system.get('last_server_sync', '-')}",
                f"Exchange: {exchange_status}",
                f"Database: {database_status}",
                f"Redis: {redis_status}",
                f"Latency: {system.get('api_latency_ms', '-') } ms",
            ]
        )
