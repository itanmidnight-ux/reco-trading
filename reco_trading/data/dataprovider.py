"""
DataProvider for Reco-Trading.
Provides data to strategies including OHLCV, ticker, and orderbook data.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from pandas import DataFrame

from reco_trading.constants import Config
from reco_trading.data.cache import MarketDataCache, get_market_cache


logger = logging.getLogger(__name__)

MAX_DATAFRAME_CANDLES = 1000

DEFAULT_OHLCV_TTL = 30.0
DEFAULT_TICKER_TTL = 5.0
DEFAULT_ORDERBOOK_TTL = 3.0


class DataProvider:
    """
    DataProvider provides data to strategies.
    It gives access to historical candles, live ticker data, and orderbook data.
    Supports caching and pre-fetching for improved performance.
    """
    
    def __init__(
        self,
        config: Config,
        exchange: Any = None,
        pairlists: Any = None,
        enable_cache: bool = True,
        enable_prefetch: bool = True,
    ) -> None:
        """
        Initialize DataProvider.
        
        Args:
            config: Configuration dictionary
            exchange: Exchange instance
            pairlists: PairList manager instance
            enable_cache: Enable in-memory caching
            enable_prefetch: Enable background pre-fetching
        """
        self._config = config
        self._exchange = exchange
        self._pairlists = pairlists
        self._cached_pairs: dict[tuple[str, str], tuple[DataFrame, datetime]] = {}
        self._default_timeframe = config.get("timeframe", "5m")
        self._enable_cache = enable_cache
        self._enable_prefetch = enable_prefetch
        self._cache = get_market_cache() if enable_cache else None
        self._prefetch_task: asyncio.Task | None = None
        self._prefetch_symbols: set[str] = set()
        self._prefetch_timeframes: list[str] = []
        
    @property
    def default_timeframe(self) -> str:
        """Get default timeframe."""
        return self._default_timeframe
    
    @property
    def config(self) -> Config:
        """Get configuration."""
        return self._config
    
    def get_pairs(self) -> list[str]:
        """
        Get currently active pairs.
        
        Returns:
            List of active trading pairs
        """
        if self._pairlists:
            return self._pairlists.get_pairlist()
        return self._config.get("exchange", {}).get("pair_whitelist", [])
    
    def get_available_pairs(self) -> list[tuple[str, str]]:
        """
        Get all available pairs with timeframes.
        
        Returns:
            List of (pair, timeframe) tuples
        """
        pairs = self.get_pairs()
        return [(pair, self._default_timeframe) for pair in pairs]
    
    def get_data(self, pair: str, timeframe: str | None = None) -> DataFrame | None:
        """
        Get cached OHLCV data for a pair.
        
        Args:
            pair: Trading pair (e.g., "BTC/USDT")
            timeframe: Timeframe (defaults to default_timeframe)
            
        Returns:
            DataFrame with OHLCV data or None if not available
        """
        if timeframe is None:
            timeframe = self._default_timeframe
            
        key = (pair, timeframe)
        
        if key in self._cached_pairs:
            return self._cached_pairs[key][0]
        
        return None
    
    def _set_cached_data(
        self, 
        pair: str, 
        timeframe: str, 
        dataframe: DataFrame
    ) -> None:
        """
        Store cached OHLCV data.
        
        Args:
            pair: Trading pair
            timeframe: Timeframe
            dataframe: OHLCV DataFrame
        """
        key = (pair, timeframe)
        self._cached_pairs[key] = (dataframe, datetime.now())
    
    async def get_historic_data(
        self, 
        pair: str, 
        timeframe: str | None = None,
        limit: int | None = None
    ) -> DataFrame | None:
        """
        Fetch historical OHLCV data from exchange.
        
        Args:
            pair: Trading pair
            timeframe: Timeframe
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data or None
        """
        if timeframe is None:
            timeframe = self._default_timeframe
            
        if limit is None:
            limit = self._config.get("history_limit", 300)
        
        if self._exchange:
            try:
                data = await self._exchange.fetch_ohlcv(
                    pair, 
                    timeframe=timeframe, 
                    limit=limit
                )
                
                if data:
                    df = self._ohlcv_to_dataframe(data)
                    self._set_cached_data(pair, timeframe, df)
                    return df
            except Exception as e:
                logger.error(f"Error fetching historic data for {pair}: {e}")
        
        return None
    
    def _ohlcv_to_dataframe(self, ohlcv: list) -> DataFrame:
        """
        Convert OHLCV data to DataFrame.
        
        Args:
            ohlcv: List of OHLCV data
            
        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        import pandas as pd
        
        df = pd.DataFrame(
            ohlcv, 
            columns=["date", "open", "high", "low", "close", "volume"]
        )
        df["date"] = pd.to_datetime(df["date"], unit="ms")
        
        return df
    
    async def get_ticker(self, pair: str) -> dict | None:
        """
        Get current ticker data for a pair.
        
        Args:
            pair: Trading pair
            
        Returns:
            Ticker data dictionary or None
        """
        if self._exchange:
            try:
                return await self._exchange.fetch_ticker(pair)
            except Exception as e:
                logger.error(f"Error fetching ticker for {pair}: {e}")
        
        return None
    
    async def get_orderbook(self, pair: str, limit: int = 20) -> dict | None:
        """
        Get orderbook data for a pair.
        
        Args:
            pair: Trading pair
            limit: Orderbook depth
            
        Returns:
            Orderbook dictionary or None
        """
        if self._exchange:
            try:
                return await self._exchange.fetch_order_book(pair, limit)
            except Exception as e:
                logger.error(f"Error fetching orderbook for {pair}: {e}")
        
        return None
    
    async def get_balance(self) -> dict | None:
        """
        Get account balance.
        
        Returns:
            Balance dictionary or None
        """
        if self._exchange:
            try:
                return await self._exchange.fetch_balance()
            except Exception as e:
                logger.error(f"Error fetching balance: {e}")
        
        return None
    
    def get_whitelist(self) -> list[str]:
        """
        Get current pair whitelist.
        
        Returns:
            List of whitelisted pairs
        """
        return self.get_pairs()
    
    def get_blacklist(self) -> list[str]:
        """
        Get current pair blacklist.
        
        Returns:
            List of blacklisted pairs
        """
        return self._config.get("exchange", {}).get("pair_blacklist", [])
    
    def get_fiat_display_currency(self) -> str:
        """
        Get fiat display currency.
        
        Returns:
            Fiat currency code
        """
        return self._config.get("fiat_display_currency", "USD")
    
    async def start_prefetch(self, symbols: list[str], timeframes: list[str]) -> None:
        """Start background pre-fetching for given symbols and timeframes."""
        if not self._enable_prefetch or not self._exchange:
            return
        self._prefetch_symbols = set(symbols)
        self._prefetch_timeframes = timeframes
        if self._prefetch_task is None or self._prefetch_task.done():
            self._prefetch_task = asyncio.create_task(self._prefetch_loop())

    async def _prefetch_loop(self) -> None:
        """Background loop that pre-fetches market data."""
        while self._prefetch_symbols and self._exchange:
            try:
                await self._prefetch_all()
                await asyncio.sleep(25.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Prefetch error: {e}")
                await asyncio.sleep(10.0)

    async def _prefetch_all(self) -> None:
        """Pre-fetch data for all configured symbols and timeframes."""
        if not self._exchange:
            return
        tasks = []
        for symbol in self._prefetch_symbols:
            for timeframe in self._prefetch_timeframes:
                tasks.append(self._prefetch_symbol_timeframe(symbol, timeframe))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _prefetch_symbol_timeframe(self, symbol: str, timeframe: str) -> None:
        """Pre-fetch data for a single symbol/timeframe."""
        if not self._exchange:
            return
        try:
            limit = self._config.get("history_limit", 300)
            data = await self._exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if data:
                df = self._ohlcv_to_dataframe(data)
                if self._cache:
                    self._cache.set_ohlcv(symbol, timeframe, df, ttl=DEFAULT_OHLCV_TTL)
        except Exception as e:
            logger.debug(f"Prefetch {symbol} {timeframe}: {e}")

    async def stop_prefetch(self) -> None:
        """Stop background pre-fetching."""
        if self._prefetch_task and not self._prefetch_task.done():
            self._prefetch_task.cancel()
            try:
                await self._prefetch_task
            except asyncio.CancelledError:
                pass

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        if self._cache:
            stats = self._cache.stats
            return {
                "hits": stats.hits,
                "misses": stats.misses,
                "hit_rate": stats.hit_rate,
                "evictions": stats.evictions,
            }
        return {}

    def refresh_data(self) -> None:
        """Refresh cached data."""
        self._cached_pairs.clear()
        if self._cache:
            self._cache.clear()

    def invalidate_symbol(self, symbol: str) -> None:
        """Invalidate all cached data for a symbol."""
        if self._cache:
            self._cache.invalidate_symbol(symbol)

    def __repr__(self) -> str:
        return f"DataProvider(exchange={self._exchange}, pairs={self.get_pairs()})"
