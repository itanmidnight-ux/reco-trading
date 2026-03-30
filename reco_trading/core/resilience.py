from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ResilienceConfig:
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    exponential_backoff: bool = True
    max_backoff_seconds: float = 60.0
    health_check_interval_seconds: float = 30.0
    graceful_shutdown_timeout_seconds: float = 30.0
    crash_recovery_enabled: bool = True
    auto_restart_on_crash: bool = True
    state_persistence_interval_seconds: float = 60.0
    max_consecutive_failures: int = 5
    network_timeout_seconds: float = 30.0


@dataclass
class FailureRecord:
    timestamp: datetime
    failure_type: str
    error_message: str
    recovery_time_seconds: float | None = None


@dataclass
class SystemState:
    last_heartbeat: datetime | None = None
    consecutive_failures: int = 0
    total_failures: int = 0
    last_successful_trade: datetime | None = None
    last_recovery: datetime | None = None
    is_healthy: bool = True
    state_file_path: str = "./user_data/bot_state.json"


class ResilienceManager:
    """
    Manages system resilience including crash recovery, network handling, and auto-restart.
    """

    def __init__(self, config: ResilienceConfig | None = None) -> None:
        self.config = config or ResilienceConfig()
        self.logger = logging.getLogger(__name__)
        self._failure_history: list[FailureRecord] = []
        self._state = SystemState()
        self._shutdown_event = asyncio.Event()
        self._restart_requested = False
        self._health_check_task: asyncio.Task | None = None
        self._state_persistence_task: asyncio.Task | None = None
        self._callbacks: dict[str, Callable] = {}
        
        self._load_state()

    def register_callback(self, event: str, callback: Callable) -> None:
        self._callbacks[event] = callback

    async def start(self) -> None:
        self.logger.info("Resilience manager started")
        self._state.last_heartbeat = datetime.now(timezone.utc)
        
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._state_persistence_task = asyncio.create_task(self._state_persistence_loop())
        
        self._setup_signal_handlers()

    async def stop(self) -> None:
        self.logger.info("Resilience manager stopping")
        self._shutdown_event.set()
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
                
        if self._state_persistence_task:
            self._state_persistence_task.cancel()
            try:
                await self._state_persistence_task
            except asyncio.CancelledError:
                pass
        
        self._save_state()
        self.logger.info("Resilience manager stopped")

    def _setup_signal_handlers(self) -> None:
        def handle_signal(sig):
            self.logger.warning(f"Received signal {sig}, initiating graceful shutdown")
            self._shutdown_event.set()
            
        try:
            signal.signal(signal.SIGTERM, lambda s, f: handle_signal(s))
            signal.signal(signal.SIGINT, lambda s, f: handle_signal(s))
        except (ValueError, OSError):
            self.logger.warning("Could not set signal handlers (non-UNIX system)")

    async def _health_check_loop(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.health_check_interval_seconds)
                self._state.last_heartbeat = datetime.now(timezone.utc)
                
                if self._state.consecutive_failures >= self.config.max_consecutive_failures:
                    self._state.is_healthy = False
                    self.logger.error(
                        f"System unhealthy: {self._state.consecutive_failures} consecutive failures"
                    )
                    if self.config.auto_restart_on_crash:
                        await self._trigger_restart("consecutive_failures")
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.logger.error(f"Health check error: {exc}")

    async def _state_persistence_loop(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.state_persistence_interval_seconds)
                self._save_state()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.logger.error(f"State persistence error: {exc}")

    def record_failure(self, failure_type: str, error_message: str) -> None:
        record = FailureRecord(
            timestamp=datetime.now(timezone.utc),
            failure_type=failure_type,
            error_message=error_message,
        )
        self._failure_history.append(record)
        self._state.consecutive_failures += 1
        self._state.total_failures += 1
        
        if len(self._failure_history) > 1000:
            self._failure_history = self._failure_history[-500:]
        
        self.logger.warning(
            f"Failure recorded: {failure_type} - {error_message} "
            f"(consecutive: {self._state.consecutive_failures})"
        )
        
        callback = self._callbacks.get("on_failure")
        if callback:
            asyncio.create_task(callback(record))

    def record_success(self) -> None:
        if self._state.consecutive_failures > 0:
            self.logger.info(
                f"Recovery successful after {self._state.consecutive_failures} failures"
            )
        self._state.consecutive_failures = 0
        self._state.is_healthy = True
        self._state.last_recovery = datetime.now(timezone.utc)
        self._state.last_successful_trade = datetime.now(timezone.utc)

    async def _trigger_restart(self, reason: str) -> None:
        if self._restart_requested:
            return
            
        self._restart_requested = True
        self.logger.error(f"Triggering auto-restart: {reason}")
        
        callback = self._callbacks.get("on_restart")
        if callback:
            try:
                await callback(reason)
            except Exception as exc:
                self.logger.error(f"Restart callback error: {exc}")

    def request_restart(self) -> None:
        self._restart_requested = True
        self._shutdown_event.set()

    def should_restart(self) -> bool:
        return self._restart_requested

    async def with_retry(
        self,
        operation: Callable,
        operation_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        last_exception: Exception | None = None
        
        for attempt in range(self.config.max_retries):
            try:
                result = await operation(*args, **kwargs)
                self.record_success()
                return result
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_exception = exc
                delay = self._calculate_delay(attempt)
                
                self.record_failure(
                    failure_type=operation_name,
                    error_message=str(exc),
                )
                
                self.logger.warning(
                    f"Operation {operation_name} failed (attempt {attempt + 1}/{self.config.max_retries}): {exc}. "
                    f"Retrying in {delay:.1f}s..."
                )
                
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(delay)
        
        raise last_exception or RuntimeError(f"Operation {operation_name} failed after {self.config.max_retries} attempts")

    def _calculate_delay(self, attempt: int) -> float:
        if self.config.exponential_backoff:
            delay = self.config.retry_delay_seconds * (2 ** attempt)
            return min(delay, self.config.max_backoff_seconds)
        return self.config.retry_delay_seconds

    def _load_state(self) -> None:
        try:
            state_file = Path(self._state.state_file_path)
            if state_file.exists():
                import json
                with open(state_file) as f:
                    data = json.load(f)
                    self._state.consecutive_failures = data.get("consecutive_failures", 0)
                    self._state.total_failures = data.get("total_failures", 0)
                    self._state.is_healthy = data.get("is_healthy", True)
                self.logger.info("Loaded resilience state")
        except Exception as exc:
            self.logger.warning(f"Could not load state: {exc}")

    def _save_state(self) -> None:
        try:
            import json
            state_file = Path(self._state.state_file_path)
            state_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "consecutive_failures": self._state.consecutive_failures,
                "total_failures": self._state.total_failures,
                "is_healthy": self._state.is_healthy,
                "last_heartbeat": self._state.last_heartbeat.isoformat() if self._state.last_heartbeat else None,
                "last_recovery": self._state.last_recovery.isoformat() if self._state.last_recovery else None,
            }
            
            with open(state_file, "w") as f:
                json.dump(data, f)
        except Exception as exc:
            self.logger.warning(f"Could not save state: {exc}")

    def get_failure_history(self, limit: int = 10) -> list[FailureRecord]:
        return self._failure_history[-limit:]

    def get_health_status(self) -> dict[str, Any]:
        return {
            "is_healthy": self._state.is_healthy,
            "consecutive_failures": self._state.consecutive_failures,
            "total_failures": self._state.total_failures,
            "last_heartbeat": self._state.last_heartbeat,
            "last_recovery": self._state.last_recovery,
            "last_successful_trade": self._state.last_successful_trade,
        }


class NetworkResilience:
    """Handles network-specific resilience issues."""

    def __init__(self, timeout_seconds: float = 30.0, max_retries: int = 3) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)
        self._connection_lost_count = 0

    async def with_network_retry(
        self,
        operation: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        last_exception: Exception | None = None
        
        for attempt in range(self.max_retries):
            try:
                return await asyncio.wait_for(
                    operation(*args, **kwargs),
                    timeout=self.timeout_seconds
                )
            except asyncio.TimeoutError:
                last_exception = RuntimeError(f"Operation timed out after {self.timeout_seconds}s")
                self.logger.warning(f"Network timeout (attempt {attempt + 1}/{self.max_retries})")
            except (ConnectionError, OSError) as exc:
                last_exception = exc
                self._connection_lost_count += 1
                self.logger.warning(f"Connection error (attempt {attempt + 1}/{self.max_retries}): {exc}")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_exception = exc
                self.logger.warning(f"Network operation failed (attempt {attempt + 1}/{self.max_retries}): {exc}")
            
            if attempt < self.max_retries - 1:
                await asyncio.sleep(min(2 ** attempt * 1.0, 10.0))
        
        raise last_exception or RuntimeError("Network operation failed")

    def get_connection_status(self) -> dict[str, Any]:
        return {
            "connection_lost_count": self._connection_lost_count,
            "timeout_seconds": self.timeout_seconds,
        }
