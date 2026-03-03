from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress


class TimeSyncService:
    def __init__(self, fetch_server_time_ms: Callable[[], Awaitable[int]], poll_interval_seconds: float = 30.0, ewma_alpha: float = 0.2) -> None:
        self._fetch_server_time_ms = fetch_server_time_ms
        self._poll_interval_seconds = max(float(poll_interval_seconds), 1.0)
        self._ewma_alpha = min(max(float(ewma_alpha), 0.01), 1.0)
        self._offset_ms = 0.0
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._sync_lock = asyncio.Lock()

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name='time-sync-service')

    async def _run(self) -> None:
        await self.force_resync()
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._poll_interval_seconds)
            except asyncio.TimeoutError:
                await self.force_resync()

    async def force_resync(self) -> None:
        async with self._sync_lock:
            server_time_ms = int(await self._fetch_server_time_ms())
            local_time_ms = int(time.time() * 1000)
            offset = float(server_time_ms - local_time_ms)
            if self._offset_ms == 0.0:
                self._offset_ms = offset
            else:
                self._offset_ms = (self._ewma_alpha * offset) + ((1.0 - self._ewma_alpha) * self._offset_ms)

    def get_corrected_timestamp_ms(self) -> int:
        return int((time.time() * 1000) + self._offset_ms)

    async def close(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None
