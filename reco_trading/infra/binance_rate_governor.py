from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Mapping


class _TokenBucket:
    def __init__(self, capacity: int, refill_seconds: float) -> None:
        self.capacity = max(float(capacity), 1.0)
        self.refill_seconds = max(float(refill_seconds), 0.1)
        self.tokens = self.capacity
        self.last_refill_monotonic = time.monotonic()

    def _refill(self, now_mono: float) -> None:
        elapsed = max(now_mono - self.last_refill_monotonic, 0.0)
        rate = self.capacity / self.refill_seconds
        self.tokens = min(self.capacity, self.tokens + (elapsed * rate))
        self.last_refill_monotonic = now_mono

    def can_consume(self, amount: float, now_mono: float) -> bool:
        self._refill(now_mono)
        return self.tokens >= amount

    def consume(self, amount: float, now_mono: float) -> None:
        self._refill(now_mono)
        self.tokens = max(self.tokens - amount, 0.0)

    def estimate_wait_seconds(self, amount: float, now_mono: float) -> float:
        self._refill(now_mono)
        if self.tokens >= amount:
            return 0.0
        deficit = amount - self.tokens
        refill_rate = self.capacity / self.refill_seconds
        return deficit / max(refill_rate, 1e-9)


class BinanceRateGovernor:
    PRIORITY_ORDER = 0
    PRIORITY_ACCOUNT = 1
    PRIORITY_MARKET_DATA = 2
    PRIORITY_TELEMETRY = 3

    def __init__(
        self,
        request_weight_1m_limit: int = 1200,
        order_count_10s_limit: int = 100,
        order_count_1m_limit: int = 1000,
    ) -> None:
        self._weight_bucket = _TokenBucket(capacity=request_weight_1m_limit, refill_seconds=60.0)
        self._order_10s_bucket = _TokenBucket(capacity=order_count_10s_limit, refill_seconds=10.0)
        self._order_1m_bucket = _TokenBucket(capacity=order_count_1m_limit, refill_seconds=60.0)
        self._lock = asyncio.Lock()
        self._cooldown_until_monotonic = 0.0
        self._cooldown_level = 0

    async def acquire(self, route_type: str, weight: int, priority: int) -> None:
        weight_cost = max(int(weight), 1)
        normalized_type = str(route_type).lower()
        while True:
            sleep_seconds = 0.0
            async with self._lock:
                now_mono = time.monotonic()
                if now_mono < self._cooldown_until_monotonic:
                    if priority >= self.PRIORITY_MARKET_DATA:
                        sleep_seconds = self._cooldown_until_monotonic - now_mono
                    else:
                        sleep_seconds = max((self._cooldown_until_monotonic - now_mono) * 0.25, 0.05)
                else:
                    waits = [self._weight_bucket.estimate_wait_seconds(float(weight_cost), now_mono)]
                    if normalized_type == 'order':
                        waits.append(self._order_10s_bucket.estimate_wait_seconds(1.0, now_mono))
                        waits.append(self._order_1m_bucket.estimate_wait_seconds(1.0, now_mono))
                    sleep_seconds = max(waits)

                    if sleep_seconds <= 0.0:
                        self._weight_bucket.consume(float(weight_cost), now_mono)
                        if normalized_type == 'order':
                            self._order_10s_bucket.consume(1.0, now_mono)
                            self._order_1m_bucket.consume(1.0, now_mono)
                        return
            await asyncio.sleep(max(sleep_seconds, 0.01))

    def observe_headers(self, headers: Mapping[str, str] | None) -> None:
        if not headers:
            return
        now_mono = time.monotonic()
        used_weight_1m = _as_int(headers.get('X-MBX-USED-WEIGHT-1M'))
        used_order_10s = _as_int(headers.get('X-MBX-ORDER-COUNT-10S'))
        used_order_1m = _as_int(headers.get('X-MBX-ORDER-COUNT-1M'))

        if used_weight_1m is not None:
            self._weight_bucket.tokens = max(self._weight_bucket.capacity - float(used_weight_1m), 0.0)
            self._weight_bucket.last_refill_monotonic = now_mono
        if used_order_10s is not None:
            self._order_10s_bucket.tokens = max(self._order_10s_bucket.capacity - float(used_order_10s), 0.0)
            self._order_10s_bucket.last_refill_monotonic = now_mono
        if used_order_1m is not None:
            self._order_1m_bucket.tokens = max(self._order_1m_bucket.capacity - float(used_order_1m), 0.0)
            self._order_1m_bucket.last_refill_monotonic = now_mono

    def enter_cooldown(self, attempt: int) -> float:
        jitter = random.uniform(0.0, 0.25)
        backoff = min((2 ** max(attempt, 1)) + jitter, 60.0)
        self._cooldown_level = max(self._cooldown_level, attempt)
        self._cooldown_until_monotonic = max(self._cooldown_until_monotonic, time.monotonic() + backoff)
        return backoff


def _as_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
