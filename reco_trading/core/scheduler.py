from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import logging


@dataclass(slots=True)
class ScheduledTask:
    name: str
    interval_seconds: float
    coroutine: Callable[[], Awaitable[None]]


class EngineScheduler:
    """Async scheduler to orchestrate periodic engine tasks."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self._tasks: list[asyncio.Task[None]] = []
        self._stop = asyncio.Event()

    def register(self, task: ScheduledTask) -> None:
        self._tasks.append(asyncio.create_task(self._runner(task), name=f"sched-{task.name}"))

    async def _runner(self, task: ScheduledTask) -> None:
        while not self._stop.is_set():
            try:
                await task.coroutine()
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("scheduled_task_failed task=%s error=%s", task.name, exc)
            await asyncio.sleep(max(task.interval_seconds, 0.05))

    async def shutdown(self) -> None:
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
