from __future__ import annotations

import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class RuntimeStatus:
    started_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    bot_status: str = "INITIALIZING"
    restart_count: int = 0
    consecutive_failures: int = 0
    manual_pause: bool = False
    kill_switch: bool = False
    exchange_pause_until: str | None = None
    open_positions: int = 0
    snapshot: dict[str, Any] = field(default_factory=dict)


class RuntimeControl:
    """Cross-thread control plane for bot runtime and remote API."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._commands: list[dict[str, Any]] = []
        self._status = RuntimeStatus()

    def enqueue(self, action: str, **payload: Any) -> None:
        with self._lock:
            self._commands.append({"action": action, "payload": payload, "ts": time.time()})

    def pop_commands(self) -> list[dict[str, Any]]:
        with self._lock:
            pending = list(self._commands)
            self._commands.clear()
        return pending

    def mark_restart(self, consecutive_failures: int) -> None:
        with self._lock:
            self._status.restart_count += 1
            self._status.consecutive_failures = consecutive_failures

    def heartbeat(self, bot_status: str, snapshot: dict[str, Any], open_positions: int, exchange_pause_until: datetime | None = None) -> None:
        with self._lock:
            self._status.last_heartbeat = time.time()
            self._status.bot_status = bot_status
            self._status.open_positions = int(open_positions)
            self._status.exchange_pause_until = (
                exchange_pause_until.astimezone(timezone.utc).isoformat(timespec="seconds") if exchange_pause_until else None
            )
            self._status.snapshot = deepcopy(snapshot)

    def set_manual_pause(self, paused: bool) -> None:
        with self._lock:
            self._status.manual_pause = paused

    def activate_kill_switch(self) -> None:
        with self._lock:
            self._status.kill_switch = True

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            uptime = max(time.time() - self._status.started_at, 0.0)
            heartbeat_age = max(time.time() - self._status.last_heartbeat, 0.0)
            return {
                "uptime_seconds": uptime,
                "heartbeat_age_seconds": heartbeat_age,
                "bot_status": self._status.bot_status,
                "restart_count": self._status.restart_count,
                "consecutive_failures": self._status.consecutive_failures,
                "manual_pause": self._status.manual_pause,
                "kill_switch": self._status.kill_switch,
                "exchange_pause_until": self._status.exchange_pause_until,
                "open_positions": self._status.open_positions,
                "snapshot": deepcopy(self._status.snapshot),
            }
