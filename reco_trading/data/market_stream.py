from __future__ import annotations

import logging

import pandas as pd

from reco_trading.data.cache import get_market_cache, DEFAULT_OHLCV_TTL
from reco_trading.data.candle_builder import ohlcv_to_frame
from reco_trading.exchange.binance_client import BinanceClient


class MarketStream:
    """Market data polling service with caching support."""

    def __init__(self, client: BinanceClient, symbol: str, limit: int, use_cache: bool = True) -> None:
        self.client = client
        self.symbol = symbol
        self.limit = limit
        self.use_cache = use_cache
        self.cache = get_market_cache() if use_cache else None
        self.logger = logging.getLogger(__name__)

    async def fetch_frame(self, timeframe: str) -> pd.DataFrame:
        if self.cache:
            cached, hit = self.cache.get_ohlcv(self.symbol, timeframe)
            if hit and cached is not None:
                return cached

        ohlcv = await self.client.fetch_ohlcv(self.symbol, timeframe, self.limit)
        df = ohlcv_to_frame(ohlcv)

        if self.cache and not df.empty:
            self.cache.set_ohlcv(self.symbol, timeframe, df, DEFAULT_OHLCV_TTL)

        return df

    async def fetch_frame_with_ttl(self, timeframe: str, ttl: float) -> pd.DataFrame:
        """Fetch frame with custom TTL."""
        if self.cache:
            cached, hit = self.cache.get_ohlcv(self.symbol, timeframe)
            if hit and cached is not None:
                return cached

        ohlcv = await self.client.fetch_ohlcv(self.symbol, timeframe, self.limit)
        df = ohlcv_to_frame(ohlcv)

        if self.cache and not df.empty:
            self.cache.set_ohlcv(self.symbol, timeframe, df, ttl)

        return df

    def invalidate_cache(self) -> None:
        """Invalidate all cached data for this symbol."""
        if self.cache:
            self.cache.invalidate_symbol(self.symbol)

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        if self.cache:
            stats = self.cache.stats
            return {
                "hits": stats.hits,
                "misses": stats.misses,
                "hit_rate": stats.hit_rate,
                "size": self.cache.size,
            }
        return {}
