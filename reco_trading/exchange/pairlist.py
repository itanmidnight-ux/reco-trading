from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PairListConfig:
    """Configuration for pairlist handlers."""
    exchange: Any = None
    config: dict = field(default_factory=dict)
    pairlist_cache: dict = field(default_factory=dict)
    blacklist: list[str] = field(default_factory=list)
    whitelist: list[str] = field(default_factory=list)


class IPairListHandler(ABC):
    """Base interface for pairlist handlers."""

    def __init__(self, exchange: Any, config: dict, pairlist_cache: dict, blacklist: list[str]):
        self.exchange = exchange
        self.config = config
        self.pairlist_cache = pairlist_cache
        self.blacklist = blacklist

    @abstractmethod
    def filter_pairlist(self, pairlist: list[str], tickers: dict) -> list[str]:
        """Filter and return the final pairlist."""
        pass

    def _validate_pair(self, pair: str, tickers: dict) -> bool:
        """Validate if a pair has required data."""
        if pair in self.blacklist:
            return False
        if pair not in tickers:
            return False
        return True


class StaticPairList(IPairListHandler):
    """Static whitelist of pairs."""

    def filter_pairlist(self, pairlist: list[str], tickers: dict) -> list[str]:
        pairs = self.config.get("pairs", pairlist)
        return [p for p in pairs if self._validate_pair(p, tickers)]


class VolumePairList(IPairListHandler):
    """Filter pairs by trading volume."""

    def filter_pairlist(self, pairlist: list[str], tickers: dict) -> list[str]:
        min_volume = self.config.get("min_volume", 1000000)
        top_n = self.config.get("top_n", 20)
        
        valid_pairs = []
        for pair in pairlist:
            if not self._validate_pair(pair, tickers):
                continue
            
            ticker = tickers.get(pair, {})
            quote_volume = float(ticker.get("quoteVolume", 0))
            
            if quote_volume >= min_volume:
                valid_pairs.append((pair, quote_volume))
        
        valid_pairs.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in valid_pairs[:top_n]]


class PercentChangePairList(IPairListHandler):
    """Filter pairs by price percent change."""

    def filter_pairlist(self, pairlist: list[str], tickers: dict) -> list[str]:
        lookback_days = self.config.get("lookback_days", 7)
        lookback_hours = self.config.get("lookback_hours", 0)
        min_change = self.config.get("min_change", -10)
        max_change = self.config.get("max_change", 10)
        top_n = self.config.get("top_n", 20)
        
        valid_pairs = []
        for pair in pairlist:
            if not self._validate_pair(pair, tickers):
                continue
            
            ticker = tickers.get(pair, {})
            change_24h = float(ticker.get("percentage", 0))
            
            if min_change <= change_24h <= max_change:
                valid_pairs.append((pair, abs(change_24h)))
        
        valid_pairs.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in valid_pairs[:top_n]]


class PriceFilter(IPairListHandler):
    """Filter pairs by price range."""

    def filter_pairlist(self, pairlist: list[str], tickers: dict) -> list[str]:
        min_price = self.config.get("min_price", 0.0001)
        max_price = self.config.get("max_price", 1000000)
        
        valid_pairs = []
        for pair in pairlist:
            if not self._validate_pair(pair, tickers):
                continue
            
            ticker = tickers.get(pair, {})
            price = float(ticker.get("last", 0))
            
            if min_price <= price <= max_price:
                valid_pairs.append(pair)
        
        return valid_pairs


class AgeFilter(IPairListHandler):
    """Filter pairs by coin age (new coins have higher risk)."""

    def filter_pairlist(self, pairlist: list[str], tickers: dict) -> list[str]:
        min_age_days = self.config.get("min_age_days", 30)
        
        try:
            from reco_trading.data.dataprovider import DataProvider
            dp = DataProvider(self.exchange)
        except Exception:
            logger.warning("AgeFilter requires DataProvider, skipping age filter")
            return pairlist
        
        now = datetime.now(timezone.utc)
        valid_pairs = []
        
        for pair in pairlist:
            if not self._validate_pair(pair, tickers):
                continue
            
            try:
                ohlcv = dp.ohlcv(pair, "1d", 1)
                if ohlcv and len(ohlcv) > 0:
                    first_candle_time = datetime.fromtimestamp(ohlcv[0][0] / 1000, tz=timezone.utc)
                    age_days = (now - first_candle_time).days
                    
                    if age_days >= min_age_days:
                        valid_pairs.append(pair)
            except Exception:
                valid_pairs.append(pair)
        
        return valid_pairs


class PrecisionFilter(IPairListHandler):
    """Filter pairs by price precision."""

    def filter_pairlist(self, pairlist: list[str], tickers: dict) -> list[str]:
        max_precision = self.config.get("max_precision", 8)
        
        valid_pairs = []
        for pair in pairlist:
            if not self._validate_pair(pair, tickers):
                continue
            
            ticker = tickers.get(pair, {})
            price = float(ticker.get("last", 0))
            
            if price > 0:
                precision = len(str(price).split(".")[-1]) if "." in str(price) else 0
                if precision <= max_precision:
                    valid_pairs.append(pair)
        
        return valid_pairs


class SpreadFilter(IPairListHandler):
    """Filter pairs by bid-ask spread."""

    def filter_pairlist(self, pairlist: list[str], tickers: dict) -> list[str]:
        max_spread_percent = self.config.get("max_spread_percent", 0.5)
        
        valid_pairs = []
        for pair in pairlist:
            if not self._validate_pair(pair, tickers):
                continue
            
            ticker = tickers.get(pair, {})
            bid = float(ticker.get("bid", 0))
            ask = float(ticker.get("ask", 0))
            
            if bid > 0 and ask > 0:
                spread_percent = ((ask - bid) / ask) * 100
                if spread_percent <= max_spread_percent:
                    valid_pairs.append(pair)
        
        return valid_pairs


class VolatilityFilter(IPairListHandler):
    """Filter pairs by volatility."""

    def filter_pairlist(self, pairlist: list[str], tickers: dict) -> list[str]:
        min_volatility = self.config.get("min_volatility", 0.0)
        max_volatility = self.config.get("max_volatility", 1.0)
        
        try:
            from reco_trading.data.dataprovider import DataProvider
            dp = DataProvider(self.exchange)
        except Exception:
            logger.warning("VolatilityFilter requires DataProvider, skipping")
            return pairlist
        
        valid_pairs = []
        
        for pair in pairlist:
            if not self._validate_pair(pair, tickers):
                continue
            
            try:
                ohlcv = dp.ohlcv(pair, "1h", 168)
                if ohlcv and len(ohlcv) >= 24:
                    closes = [c[4] for c in ohlcv]
                    returns = np.diff(np.log(closes))
                    volatility = float(np.std(returns) * np.sqrt(24))
                    
                    if min_volatility <= volatility <= max_volatility:
                        valid_pairs.append(pair)
            except Exception:
                valid_pairs.append(pair)
        
        return valid_pairs


class RangeStabilityFilter(IPairListHandler):
    """Filter pairs by price range stability."""

    def filter_pairlist(self, pairlist: list[str], tickers: dict) -> list[str]:
        lookback_hours = self.config.get("lookback_hours", 24)
        min_stability = self.config.get("min_stability", 0.5)
        
        try:
            from reco_trading.data.dataprovider import DataProvider
            dp = DataProvider(self.exchange)
        except Exception:
            logger.warning("RangeStabilityFilter requires DataProvider, skipping")
            return pairlist
        
        valid_pairs = []
        
        for pair in pairlist:
            if not self._validate_pair(pair, tickers):
                continue
            
            try:
                ohlcv = dp.ohlcv(pair, "1h", lookback_hours)
                if ohlcv and len(ohlcv) >= 10:
                    highs = [c[2] for c in ohlcv]
                    lows = [c[3] for c in ohlcv]
                    
                    price_range = (max(highs) - min(lows)) / max(highs) if max(highs) > 0 else 0
                    
                    if price_range >= min_stability:
                        valid_pairs.append(pair)
            except Exception:
                valid_pairs.append(pair)
        
        return valid_pairs


class ShuffleFilter(IPairListHandler):
    """Randomly shuffle the pairlist."""

    def filter_pairlist(self, pairlist: list[str], tickers: dict) -> list[str]:
        valid_pairs = [p for p in pairlist if self._validate_pair(p, tickers)]
        seed = self.config.get("seed", None)
        
        if seed is not None:
            random.seed(seed)
        
        random.shuffle(valid_pairs)
        return valid_pairs


class OffsetFilter(IPairListHandler):
    """Skip first N pairs."""

    def filter_pairlist(self, pairlist: list[str], tickers: dict) -> list[str]:
        offset = self.config.get("offset", 0)
        limit = self.config.get("limit", None)
        
        valid_pairs = [p for p in pairlist if self._validate_pair(p, tickers)]
        
        if offset > 0:
            valid_pairs = valid_pairs[offset:]
        
        if limit:
            valid_pairs = valid_pairs[:limit]
        
        return valid_pairs


class FullTradesFilter(IPairListHandler):
    """Filter pairs that must have full candle data."""

    def filter_pairlist(self, pairlist: list[str], tickers: dict) -> list[str]:
        required_candles = self.config.get("required_candles", 300)
        
        try:
            from reco_trading.data.dataprovider import DataProvider
            dp = DataProvider(self.exchange)
        except Exception:
            logger.warning("FullTradesFilter requires DataProvider, skipping")
            return pairlist
        
        valid_pairs = []
        
        for pair in pairlist:
            if not self._validate_pair(pair, tickers):
                continue
            
            try:
                ohlcv = dp.ohlcv(pair, "5m", required_candles)
                if ohlcv and len(ohlcv) >= required_candles:
                    valid_pairs.append(pair)
            except Exception:
                pass
        
        return valid_pairs


class PerformanceFilter(IPairListHandler):
    """Filter pairs by historical performance."""

    def filter_pairlist(self, pairlist: list[str], tickers: dict) -> list[str]:
        lookback_days = self.config.get("lookback_days", 7)
        min_profit = self.config.get("min_profit", -100)
        top_n = self.config.get("top_n", 20)
        
        try:
            from reco_trading.database.repository import Repository
        except Exception:
            logger.warning("PerformanceFilter requires Repository, skipping")
            return pairlist
        
        valid_pairs = []
        
        for pair in pairlist:
            if not self._validate_pair(pair, tickers):
                continue
            
            try:
                profit = 0.0
                valid_pairs.append((pair, profit))
            except Exception:
                valid_pairs.append((pair, 0.0))
        
        valid_pairs.sort(key=lambda x: x[1], reverse=True)
        
        filtered = [p[0] for p in valid_pairs if p[1] >= min_profit]
        return filtered[:top_n]


PAIRLIST_HANDLERS = {
    "StaticPairList": StaticPairList,
    "VolumePairList": VolumePairList,
    "PercentChangePairList": PercentChangePairList,
    "PriceFilter": PriceFilter,
    "AgeFilter": AgeFilter,
    "PrecisionFilter": PrecisionFilter,
    "SpreadFilter": SpreadFilter,
    "VolatilityFilter": VolatilityFilter,
    "RangeStabilityFilter": RangeStabilityFilter,
    "ShuffleFilter": ShuffleFilter,
    "OffsetFilter": OffsetFilter,
    "FullTradesFilter": FullTradesFilter,
    "PerformanceFilter": PerformanceFilter,
}


class PairListManager:
    """Manages multiple pairlist handlers."""

    def __init__(self, exchange: Any, config: dict):
        self.logger = logging.getLogger(__name__)
        self.exchange = exchange
        self.config = config
        self.handlers: list[IPairListHandler] = []
        self.blacklist: list[str] = []
        self.pairlist_cache: dict = {}
        self._initialize_handlers()

    def _initialize_handlers(self) -> None:
        handlers_config = self.config.get("pairlist_handlers", [])
        
        for handler_config in handlers_config:
            handler_name = handler_config.get("name")
            handler_params = handler_config.get("params", {})
            
            if handler_name in PAIRLIST_HANDLERS:
                handler_class = PAIRLIST_HANDLERS[handler_name]
                handler = handler_class(
                    self.exchange,
                    handler_params,
                    self.pairlist_cache,
                    self.blacklist,
                )
                self.handlers.append(handler)
                self.logger.info(f"Initialized pairlist handler: {handler_name}")
            else:
                self.logger.warning(f"Unknown pairlist handler: {handler_name}")

    def get_pairlist(self, base_pairlist: list[str], tickers: dict) -> list[str]:
        """Get filtered pairlist through all handlers."""
        pairlist = base_pairlist
        
        for handler in self.handlers:
            pairlist = handler.filter_pairlist(pairlist, tickers)
            self.logger.info(f"After {handler.__class__.__name__}: {len(pairlist)} pairs")
        
        return pairlist

    def add_to_blacklist(self, pairs: list[str]) -> None:
        """Add pairs to blacklist."""
        self.blacklist.extend(pairs)
        self.logger.info(f"Added to blacklist: {pairs}")

    def remove_from_blacklist(self, pairs: list[str]) -> None:
        """Remove pairs from blacklist."""
        self.blacklist = [p for p in self.blacklist if p not in pairs]
        self.logger.info(f"Removed from blacklist: {pairs}")

    def get_blacklist(self) -> list[str]:
        return self.blacklist.copy()
