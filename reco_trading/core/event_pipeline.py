from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class PipelineEvent:
    topic: str
    payload: dict[str, Any]


Handler = Callable[[PipelineEvent], Awaitable[None]]


class AsyncEventBus:
    def __init__(self, maxsize: int = 1000) -> None:
        self._queue: asyncio.Queue[PipelineEvent] = asyncio.Queue(maxsize=maxsize)
        self._handlers: dict[str, list[Handler]] = {}
        self._tasks: list[asyncio.Task[None]] = []
        self._stop = asyncio.Event()

    def subscribe(self, topic: str, handler: Handler) -> None:
        self._handlers.setdefault(topic, []).append(handler)

    async def publish(self, event: PipelineEvent) -> None:
        await self._queue.put(event)

    async def _worker(self) -> None:
        while not self._stop.is_set():
            event = await self._queue.get()
            try:
                for handler in self._handlers.get(event.topic, []):
                    await handler(event)
            finally:
                self._queue.task_done()

    async def start(self, workers: int = 2) -> None:
        for _ in range(workers):
            self._tasks.append(asyncio.create_task(self._worker()))

    async def shutdown(self) -> None:
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
