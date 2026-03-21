from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QListWidget, QVBoxLayout, QWidget

from reco_trading.ui.widgets.stat_card import StatCard


class AlertsTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._last_signature: tuple[object, ...] | None = None
        self._previous_status: str | None = None
        self._previous_cooldown: str | None = None
        self._alert_history: list[str] = []

        root = QVBoxLayout(self)
        title = QLabel("Alerts Center")
        title.setObjectName("sectionTitle")
        root.addWidget(title)

        subtitle = QLabel("Alert history with status and cooldown change detection")
        subtitle.setObjectName("metricLabel")
        root.addWidget(subtitle)

        panel = QFrame()
        panel.setObjectName("panelCard")
        root.addWidget(panel)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 12, 12, 12)
        panel_layout.setSpacing(10)

        cards_layout = QGridLayout()
        self.cards = {
            "status": StatCard("Current Status", compact=True),
            "cooldown": StatCard("Cooldown", compact=True),
            "exchange": StatCard("Exchange", compact=True),
            "database": StatCard("Database", compact=True),
            "ui_lag": StatCard("UI Lag", compact=True),
            "alerts": StatCard("Tracked Alerts", compact=True),
        }
        for i, card in enumerate(self.cards.values()):
            cards_layout.addWidget(card, i // 3, i % 3)
        panel_layout.addLayout(cards_layout)

        self.summary = QLabel("Monitoring alerts...")
        self.summary.setObjectName("smallMetricValue")
        panel_layout.addWidget(self.summary)

        self.alerts_list = QListWidget()
        self.alerts_list.addItems(["No alerts yet."])
        panel_layout.addWidget(self.alerts_list)

    def update_state(self, state: dict[str, Any]) -> None:
        system = dict(state.get("system", {}) or {})
        status = str(state.get("status", "UNKNOWN"))
        cooldown = str(state.get("cooldown", "READY"))
        exchange = str(system.get("exchange_status", "UNKNOWN"))
        database = str(system.get("database_status", "UNKNOWN"))
        ui_lag = bool(system.get("ui_lag_detected", False))
        logs = state.get("logs", [])

        status_change = self._previous_status is not None and self._previous_status != status
        cooldown_change = self._previous_cooldown is not None and self._previous_cooldown != cooldown

        if self._previous_status is None:
            self._push_alert(f"Initial status detected: {status}")
        elif status_change:
            self._push_alert(f"Status changed: {self._previous_status} -> {status}")

        if self._previous_cooldown is None:
            self._push_alert(f"Initial cooldown detected: {cooldown}")
        elif cooldown_change:
            self._push_alert(f"Cooldown changed: {self._previous_cooldown} -> {cooldown}")

        self._previous_status = status
        self._previous_cooldown = cooldown

        recent_logs = tuple(
            f"[{entry.get('time', '')}] {str(entry.get('level', 'INFO')).upper()}: {entry.get('message', '')}"
            for entry in logs[-5:]
        )
        signature = (
            status,
            cooldown,
            exchange,
            database,
            ui_lag,
            tuple(self._alert_history[-20:]),
            recent_logs,
        )
        if signature == self._last_signature:
            return
        self._last_signature = signature

        self.cards["status"].set_value(status, tone=_status_tone(status))
        self.cards["cooldown"].set_value(cooldown, tone=_cooldown_tone(cooldown))
        self.cards["exchange"].set_value(exchange, tone=_service_tone(exchange))
        self.cards["database"].set_value(database, tone=_service_tone(database))
        self.cards["ui_lag"].set_value("YES" if ui_lag else "NO", tone="warning" if ui_lag else "positive")
        self.cards["alerts"].set_value(str(len(self._alert_history)), tone="info")

        self.summary.setText(
            f"Status {status} • Cooldown {cooldown} • Exchange {exchange} • Database {database} • UI lag {'ON' if ui_lag else 'OFF'}"
        )

        self.alerts_list.clear()
        items = list(self._alert_history[-30:])
        if recent_logs:
            items.extend(recent_logs)
        if not items:
            items = ["No alerts yet."]
        self.alerts_list.addItems(items)

    def _push_alert(self, message: str) -> None:
        if self._alert_history and self._alert_history[-1] == message:
            return
        self._alert_history.append(message)
        self._alert_history = self._alert_history[-200:]


def _status_tone(status: str) -> str:
    normalized = status.strip().lower()
    if normalized in {"error", "stopped"}:
        return "negative"
    if normalized in {"paused", "cooldown"}:
        return "warning"
    if normalized in {"position_open", "waiting_market_data", "analyzing_market", "signal_generated", "placing_order"}:
        return "positive"
    return "info"


def _cooldown_tone(cooldown: str) -> str:
    normalized = cooldown.strip().upper()
    if normalized in {"READY", "NONE", "-"}:
        return "positive"
    if "EMERGENCY" in normalized or "EXCHANGE" in normalized:
        return "negative"
    return "warning"


def _service_tone(value: str) -> str:
    normalized = value.strip().upper()
    if normalized in {"CONNECTED", "OK", "ONLINE"}:
        return "positive"
    if normalized in {"CONNECTING", "UNKNOWN"}:
        return "warning"
    return "negative"
