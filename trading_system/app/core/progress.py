from __future__ import annotations

import asyncio
import itertools
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass


@dataclass
class ProgressStyle:
    frames: tuple[str, ...] = ('/', '-', '\\', '|')
    interval: float = 0.12


class AsyncSpinner:
    def __init__(self, message: str, style: ProgressStyle | None = None) -> None:
        self.message = message
        self.style = style or ProgressStyle()
        self._task: asyncio.Task | None = None
        self._running = False

    async def _spin(self) -> None:
        for frame in itertools.cycle(self.style.frames):
            if not self._running:
                break
            sys.stdout.write(f'\r{self.message} {frame}')
            sys.stdout.flush()
            await asyncio.sleep(self.style.interval)

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._spin())

    async def stop(self, ok_message: str | None = None) -> None:
        self._running = False
        if self._task:
            await self._task
        sys.stdout.write('\r')
        sys.stdout.write((ok_message or f'{self.message} ✓') + '\n')
        sys.stdout.flush()


@asynccontextmanager
async def progress(message: str, done: str | None = None):
    spinner = AsyncSpinner(message)
    await spinner.start()
    try:
        yield
    finally:
        await spinner.stop(done)
