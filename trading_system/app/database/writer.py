from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

WriteCall = Callable[..., Awaitable[None]]


@dataclass
class WriteTask:
    fn: WriteCall
    kwargs: dict[str, Any]
    retries: int = 3


class AsyncDBWriter:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[WriteTask] = asyncio.Queue(maxsize=10000)

    async def submit(self, task: WriteTask) -> None:
        await self.queue.put(task)

    async def run(self) -> None:
        while True:
            task = await self.queue.get()
            try:
                for attempt in range(task.retries):
                    try:
                        await task.fn(**task.kwargs)
                        break
                    except Exception as exc:  # noqa: BLE001
                        if attempt == task.retries - 1:
                            raise
                        await asyncio.sleep(min(2.0, 0.2 * (2**attempt)))
                        logger.warning('db_write_retry', extra={'extra_payload': {'error': str(exc), 'attempt': attempt + 1}})
            except Exception as exc:  # noqa: BLE001
                logger.error('db_write_failed', extra={'extra_payload': {'error': str(exc), 'kwargs': str(task.kwargs)[:300]}})
            finally:
                self.queue.task_done()
