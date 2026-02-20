from __future__ import annotations

import asyncio
import time
from collections import deque


class AdaptiveRateLimitController:
    def __init__(self, max_calls: int = 10, period_seconds: float = 1.0) -> None:
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self._calls: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            while self._calls and now - self._calls[0] > self.period_seconds:
                self._calls.popleft()
            if len(self._calls) >= self.max_calls:
                sleep_for = self.period_seconds - (now - self._calls[0])
                await asyncio.sleep(max(sleep_for, 0.0))
            self._calls.append(time.monotonic())
