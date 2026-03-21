from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import threading
from typing import Any

from PySide6.QtCore import QObject, Signal


class StateManager(QObject):
    """Thread-safe shared state store for bot -> UI updates."""

    state_changed = Signal(dict)
    trade_added = Signal(dict)
    log_added = Signal(dict)
    notification = Signal(str, str)
    control_requested = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.RLock()
        self._control_queue: list[str] = []
        self._runtime_settings_queue: list[dict[str, Any]] = []
        self._state: dict[str, Any] = {
            "pair": "BTC/USDT",
            "timeframe": "5m / 15m",
            "status": "INITIALIZING",
            "current_price": 0.0,
            "bid": 0.0,
            "ask": 0.0,
            "spread": 0.0,
            "trend": "NEUTRAL",
            "signal": "HOLD",
            "confidence": 0.0,
            "volatility_regime": "NORMAL",
            "order_flow": "NEUTRAL",
            "adx": 0.0,
            "atr": 0.0,
            "volume": 0.0,
            "balance": 0.0,
            "equity": 0.0,
            "btc_balance": 0.0,
            "btc_value": 0.0,
            "total_equity": 0.0,
            "daily_pnl": 0.0,
            "session_pnl": 0.0,
            "trades_today": 0,
            "win_rate": 0.0,
            "has_open_position": False,
            "unrealized_pnl": 0.0,
            "open_position_side": None,
            "open_position_entry": None,
            "open_position_qty": None,
            "open_position_sl": None,
            "open_position_tp": None,
            "last_trade": "-",
            "cooldown": "READY",
            "risk_metrics": {},
            "system": {"uptime_seconds": 0.0, "api_latency_ms": 0.0, "database_status": "UNKNOWN", "exchange_status": "UNKNOWN", "redis_status": "UNKNOWN", "memory_usage_mb": 0.0, "last_server_sync": "-"},
            "signal_analysis": {},
            "trade_history": [],
            "logs": [],
            "analytics": {},
            "session_stats": {},
            "runtime_settings": {},
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._state)

    def update(self, **values: Any) -> None:
        with self._lock:
            self._state.update(values)
            payload = deepcopy(self._state)
        self.state_changed.emit(payload)

    def update_signal(self, signal_data: dict[str, Any]) -> None:
        self.update(signal_analysis=signal_data, **signal_data)

    def add_trade(self, trade: dict[str, Any]) -> None:
        with self._lock:
            history = self._state.setdefault("trade_history", [])
            history.insert(0, deepcopy(trade))
            payload = deepcopy(trade)
            state_copy = deepcopy(self._state)
        self.trade_added.emit(payload)
        self.state_changed.emit(state_copy)

    def add_log(self, level: str, message: str) -> None:
        entry = {"time": datetime.utcnow().strftime("%H:%M:%S"), "level": level.upper(), "message": message}
        with self._lock:
            logs = self._state.setdefault("logs", [])
            logs.append(entry)
            self._state["logs"] = logs[-400:]
            state_copy = deepcopy(self._state)
        self.log_added.emit(entry)
        self.state_changed.emit(state_copy)

    def notify(self, title: str, message: str) -> None:
        self.notification.emit(title, message)

    def request_start(self) -> None:
        with self._lock:
            self._control_queue.append("start")
        self.control_requested.emit("start")

    def request_pause(self) -> None:
        with self._lock:
            self._control_queue.append("pause")
        self.control_requested.emit("pause")

    def request_resume(self) -> None:
        with self._lock:
            self._control_queue.append("resume")
        self.control_requested.emit("resume")

    def request_emergency_stop(self) -> None:
        with self._lock:
            self._control_queue.append("emergency_stop")
        self.control_requested.emit("emergency_stop")

    def request_force_close(self) -> None:
        with self._lock:
            self._control_queue.append("force_close")
        self.control_requested.emit("force_close")

    def pop_control_requests(self) -> list[str]:
        with self._lock:
            pending = list(self._control_queue)
            self._control_queue.clear()
        return pending

    def push_runtime_settings(self, settings: dict[str, Any]) -> None:
        with self._lock:
            self._runtime_settings_queue.append(deepcopy(settings))

    def pop_runtime_settings(self) -> list[dict[str, Any]]:
        with self._lock:
            pending = list(self._runtime_settings_queue)
            self._runtime_settings_queue.clear()
        return pending

    def clear_logs(self) -> None:
        with self._lock:
            self._state["logs"] = []
            state_copy = deepcopy(self._state)
        self.state_changed.emit(state_copy)
