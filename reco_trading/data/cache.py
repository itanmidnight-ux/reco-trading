from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, TypeVar

import pandas as pd

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_OHLCV_TTL = 30.0
DEFAULT_TICKER_TTL = 5.0
DEFAULT_ORDERBOOK_TTL = 3.0


@dataclass(slots=True)
class CacheEntry:
    value: Any
    timestamp: float
    ttl: float


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class DataCache:
    def __init__(self, default_ttl: float = 60.0, max_size: int = 100) -> None:
        self._cache: dict[str, CacheEntry] = {}
        self._default_ttl = default_ttl
        self._max_size = max_size
        self._lock = asyncio.Lock()
        self._stats = CacheStats()
        self._access_order: list[str] = []

    def _make_key(self, prefix: str, **kwargs) -> str:
        parts = [prefix]
        for k, v in sorted(kwargs.items()):
            if isinstance(v, pd.DataFrame):
                parts.append(f"{k}:df{len(v)}")
            else:
                parts.append(f"{k}:{v}")
        return "|".join(parts)

    def get(self, key: str) -> tuple[Any, bool]:
        if key in self._cache:
            entry = self._cache[key]
            age = time.time() - entry.timestamp
            if age < entry.ttl:
                self._stats.hits
                self._move_to_front(key)
                return entry.value, True
            else:
                del self._cache[key]
                self._stats.evictions += 1
                if key in self._access_order:
                    self._access_order.remove(key)
        self._stats.misses += 1
        return None, False

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        if len(self._cache) >= self._max_size:
            self._evict_oldest()
        ttl = ttl if ttl is not None else self._default_ttl
        self._cache[key] = CacheEntry(value=value, timestamp=time.time(), ttl=ttl)
        self._move_to_front(key)

    def _move_to_front(self, key: str) -> None:
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.insert(0, key)

    def _evict_oldest(self) -> None:
        if self._access_order:
            oldest = self._access_order.pop()
            if oldest in self._cache:
                del self._cache[oldest]
                self._stats.evictions += 1

    async def get_or_fetch(
        self,
        key: str,
        fetch_fn: Callable[[], Any],
        ttl: float | None = None,
    ) -> Any:
        value, hit = self.get(key)
        if hit:
            return value
        async with self._lock:
            value, hit = self.get(key)
            if hit:
                return value
            logger.debug(f"Cache miss, fetching: {key}")
            result = await fetch_fn()
            self.set(key, result, ttl)
            return result

    def invalidate(self, key: str) -> None:
        if key in self._cache:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)

    def invalidate_prefix(self, prefix: str) -> int:
        keys_to_remove = [k for k in self._cache.keys() if k.startswith(prefix)]
        for key in keys_to_remove:
            self.invalidate(key)
        return len(keys_to_remove)

    def clear(self) -> None:
        self._cache.clear()
        self._access_order.clear()

    @property
    def stats(self) -> CacheStats:
        return self._stats

    @property
    def size(self) -> int:
        return len(self._cache)


class MarketDataCache:
    def __init__(self, default_ttl: float = 30.0) -> None:
        self._cache = DataCache(default_ttl=default_ttl, max_size=50)
        self._ohlcv_prefix = "ohlcv"
        self._ticker_prefix = "ticker"
        self._orderbook_prefix = "ob"

    def get_ohlcv(self, symbol: str, timeframe: str) -> tuple[pd.DataFrame | None, bool]:
        key = self._cache._make_key(self._ohlcv_prefix, symbol=symbol, timeframe=timeframe)
        return self._cache.get(key)

    def set_ohlcv(self, symbol: str, timeframe: str, df: pd.DataFrame, ttl: float | None = None) -> None:
        key = self._cache._make_key(self._ohlcv_prefix, symbol=symbol, timeframe=timeframe)
        self._cache.set(key, df, ttl)

    def get_ticker(self, symbol: str) -> tuple[dict | None, bool]:
        key = self._cache._make_key(self._ticker_prefix, symbol=symbol)
        return self._cache.get(key)

    def set_ticker(self, symbol: str, data: dict, ttl: float | None = None) -> None:
        key = self._cache._make_key(self._ticker_prefix, symbol=symbol)
        self._cache.set(key, data, ttl)

    def get_orderbook(self, symbol: str, depth: int = 20) -> tuple[dict | None, bool]:
        key = self._cache._make_key(self._orderbook_prefix, symbol=symbol, depth=depth)
        return self._cache.get(key)

    def set_orderbook(self, symbol: str, depth: int, data: dict, ttl: float | None = None) -> None:
        key = self._cache._make_key(self._orderbook_prefix, symbol=symbol, depth=depth)
        self._cache.set(key, data, ttl)

    def invalidate_symbol(self, symbol: str) -> int:
        count = self._cache.invalidate_prefix(f"{self._ohlcv_prefix}|symbol:{symbol}")
        count += self._cache.invalidate_prefix(f"{self._ticker_prefix}|symbol:{symbol}")
        count += self._cache.invalidate_prefix(f"{self._orderbook_prefix}|symbol:{symbol}")
        return count

    def clear(self) -> None:
        self._cache.clear()

    @property
    def stats(self) -> CacheStats:
        return self._cache.stats


_global_cache: MarketDataCache | None = None


def get_market_cache() -> MarketDataCache:
    global _global_cache
    if _global_cache is None:
        _global_cache = MarketDataCache()
    return _global_cache
