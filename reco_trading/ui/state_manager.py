from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from queue import Empty, Queue
import threading
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal


class StateManager(QObject):
    """Thread-safe read-only snapshot bridge from engine to UI widgets."""

    state_changed = Signal(dict)
    trade_added = Signal(dict)
    log_added = Signal(dict)
    notification = Signal(str, str)

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.RLock()
        self._incoming: Queue[dict[str, Any]] = Queue(maxsize=100)
        self._state: dict[str, Any] = {
            "pair": "BTC/USDT",
            "status": "INITIALIZING",
            "current_price": 0.0,
            "spread": 0.0,
            "trend": "NEUTRAL",
            "adx": 0.0,
            "volatility_regime": "UNKNOWN",
            "order_flow": "NEUTRAL",
            "signal": "HOLD",
            "confidence": 0.0,
            "balance": 0.0,
            "equity": 0.0,
            "daily_pnl": 0.0,
            "trades_today": 0,
            "win_rate": 0.0,
            "last_trade": "--",
            "cooldown_timer": "--",
            "bot_state": "INITIALIZING",
            "risk_metrics": {},
            "system": {},
            "analytics": {},
            "trade_history": [],
            "logs": [],
        }

        self._drain_timer = QTimer(self)
        self._drain_timer.timeout.connect(self._drain_queue)
        self._drain_timer.start(120)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._state)

    def publish_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Non-blocking API for engine thread to publish a new snapshot."""
        try:
            self._incoming.put_nowait(deepcopy(snapshot))
        except Exception:
            pass

    def update(self, **values: Any) -> None:
        self.publish_snapshot(values)

    def _drain_queue(self) -> None:
        updated = False
        while True:
            try:
                payload = self._incoming.get_nowait()
            except Empty:
                break
            with self._lock:
                self._state.update(payload)
            updated = True
        if updated:
            self.state_changed.emit(self.snapshot())

    def add_trade(self, trade: dict[str, Any]) -> None:
        with self._lock:
            history = self._state.setdefault("trade_history", [])
            history.insert(0, deepcopy(trade))
            state_copy = deepcopy(self._state)
        self.trade_added.emit(deepcopy(trade))
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
