from __future__ import annotations

import platform
from typing import Any

from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QListWidget, QVBoxLayout, QWidget

from reco_trading.ui.widgets.stat_card import StatCard


class SystemTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._last_signature: tuple[object, ...] | None = None
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
            "ui_render": StatCard("UI Render", compact=True),
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
        ui_render_ms = system.get("ui_render_ms", "-")
        ui_staleness_ms = system.get("ui_staleness_ms", "-")
        ui_lag_detected = bool(system.get("ui_lag_detected", False))
        uptime_s = float(system.get("uptime_seconds", 0) or 0)
        h = int(uptime_s // 3600)
        m = int((uptime_s % 3600) // 60)
        s = int(uptime_s % 60)
        uptime_str = f"{h:02d}:{m:02d}:{s:02d}"
        signature = (
            state.get("bot_version") or "reco-trading",
            exchange_status,
            database_status,
            redis_status,
            system.get("api_latency_ms", "-"),
            system.get("memory_usage_mb", "-"),
            ui_render_ms,
            ui_staleness_ms,
            ui_lag_detected,
            system.get("uptime_seconds", 0),
            system.get("last_server_sync", "-"),
        )
        if signature == self._last_signature:
            return
        self._last_signature = signature

        self.cards["version"].set_value(str(state.get("bot_version") or "reco-trading"))
        self.cards["api"].set_value(exchange_status)
        self.cards["database"].set_value(database_status)
        self.cards["latency"].set_value(f"{system.get('api_latency_ms', '-') } ms")
        self.cards["ui_render"].set_value(f"{ui_render_ms} ms")
        mem_raw = system.get("memory_usage_mb", 0)
        try:
            mem_mb = float(mem_raw)
            mem_text = f"{mem_mb:.1f} MB"
            if mem_mb > 400:
                mem_text += " HIGH"
        except (TypeError, ValueError):
            mem_text = "-"
        self.cards["memory"].set_value(mem_text)
        self.cards["redis"].set_value(redis_status)

        if all(value in {"OK", "CONNECTED", "ONLINE", "UNKNOWN"} for value in (exchange_status, database_status, redis_status)) and not ui_lag_detected:
            self.health_badge.setText("Health: STABLE")
            self.health_badge.setStyleSheet("color:#16c784;")
        else:
            self.health_badge.setText("Health: DEGRADED")
            self.health_badge.setStyleSheet("color:#f0b90b;")

        self.events.clear()
        self.events.addItems(
            [
                f"Uptime: {uptime_str}",
                f"Last server sync: {system.get('last_server_sync', '-')}",
                f"Exchange: {exchange_status}",
                f"Database: {database_status}",
                f"Redis: {redis_status}",
                f"Latency: {system.get('api_latency_ms', '-') } ms",
                f"UI render: {ui_render_ms} ms",
                f"UI stale: {ui_staleness_ms} ms",
                f"UI lag detected: {'YES' if ui_lag_detected else 'NO'}",
            ]
        )
