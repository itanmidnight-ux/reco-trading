from __future__ import annotations

import asyncio
import time
from collections import deque


class TokenBucket:
    def __init__(self, capacity: int, refill_per_sec: float) -> None:
        self.capacity = capacity
        self.tokens = float(capacity)
        self.refill_per_sec = refill_per_sec
        self.updated_at = time.monotonic()
        self._lock = asyncio.Lock()

    async def consume(self, amount: float) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self.updated_at
                self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_sec)
                self.updated_at = now
                if self.tokens >= amount:
                    self.tokens -= amount
                    return
                wait = (amount - self.tokens) / self.refill_per_sec
            await asyncio.sleep(max(0.01, wait))


class BinanceRateLimitController:
    def __init__(self, safe_weight_limit: int = 900, order_rate_per_sec: int = 8) -> None:
        self.safe_weight_limit = safe_weight_limit
        self.weight_window: deque[tuple[float, int]] = deque()
        self.weight_lock = asyncio.Lock()
        self.order_bucket = TokenBucket(capacity=order_rate_per_sec, refill_per_sec=order_rate_per_sec)
        self.cooldown_until = 0.0

    async def reserve_weight(self, weight: int) -> None:
        while True:
            async with self.weight_lock:
                now = time.time()
                while self.weight_window and now - self.weight_window[0][0] > 60:
                    self.weight_window.popleft()
                used = sum(w for _, w in self.weight_window)
                if now < self.cooldown_until:
                    sleep_for = self.cooldown_until - now
                elif used + weight <= self.safe_weight_limit:
                    self.weight_window.append((now, weight))
                    return
                else:
                    sleep_for = 60 - (now - self.weight_window[0][0])
            await asyncio.sleep(max(0.05, sleep_for))

    async def reserve_order_slot(self) -> None:
        await self.order_bucket.consume(1)

    async def usage_1m(self) -> int:
        async with self.weight_lock:
            now = time.time()
            while self.weight_window and now - self.weight_window[0][0] > 60:
                self.weight_window.popleft()
            return sum(w for _, w in self.weight_window)

    def apply_header_usage(self, used_weight_1m: int) -> None:
        if used_weight_1m > self.safe_weight_limit:
            self.cooldown_until = time.time() + 3

    def trigger_418_cooldown(self, seconds: int = 60) -> None:
        self.cooldown_until = time.time() + seconds


async def exponential_backoff(attempt: int, base: float = 0.5, cap: float = 30.0) -> None:
    await asyncio.sleep(min(cap, base * (2**attempt)))
