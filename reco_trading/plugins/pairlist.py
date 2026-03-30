from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class PairListHandler(ABC):
    def __init__(self, config: dict[str, Any]):
        self._config = config

    @abstractmethod
    def filter_pairlist(self, pairlist: list[str], market_data: dict[str, pd.DataFrame]) -> list[str]:
        pass


class VolumePairList(PairListHandler):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._number_assets = config.get("number_assets", 10)
        self._sort_key = config.get("sort_key", "quoteVolume")
        self._refresh_period = config.get("refresh_period", 1800)

    def filter_pairlist(self, pairlist: list[str], market_data: dict[str, pd.DataFrame]) -> list[str]:
        volumes = []
        for pair in pairlist:
            if pair in market_data:
                df = market_data[pair]
                if len(df) > 0:
                    volume = df["volume"].iloc[-1] * df["close"].iloc[-1]
                    volumes.append((pair, volume))
        
        volumes.sort(key=lambda x: x[1], reverse=True)
        return [pair for pair, _ in volumes[:self._number_assets]]


class PriceFilter(PairListHandler):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._min_price = config.get("min_price", 0.0000001)
        self._max_price = config.get("max_price", None)
        self._low_price_ratio = config.get("low_price_ratio", 0.0001)

    def filter_pairlist(self, pairlist: list[str], market_data: dict[str, pd.DataFrame]) -> list[str]:
        filtered = []
        for pair in pairlist:
            if pair in market_data:
                df = market_data[pair]
                if len(df) > 0:
                    price = df["close"].iloc[-1]
                    if price >= self._min_price:
                        if self._max_price is None or price <= self._max_price:
                            filtered.append(pair)
        return filtered


class SpreadFilter(PairListHandler):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._max_spread_ratio = config.get("max_spread_ratio", 0.005)

    def filter_pairlist(self, pairlist: list[str], market_data: dict[str, pd.DataFrame]) -> list[str]:
        filtered = []
        for pair in pairlist:
            if pair in market_data:
                df = market_data[pair]
                if len(df) > 0:
                    bid = df["low"].iloc[-1]
                    ask = df["high"].iloc[-1]
                    if bid > 0:
                        spread = (ask - bid) / bid
                        if spread <= self._max_spread_ratio:
                            filtered.append(pair)
        return filtered


class VolatilityFilter(PairListHandler):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._min_volatility = config.get("min_volatility", 0.0)
        self._max_volatility = config.get("max_volatility", 0.15)
        self._lookback_days = config.get("lookback_days", 10)

    def filter_pairlist(self, pairlist: list[str], market_data: dict[str, pd.DataFrame]) -> list[str]:
        filtered = []
        for pair in pairlist:
            if pair in market_data:
                df = market_data[pair]
                if len(df) >= self._lookback_days:
                    returns = df["close"].pct_change().dropna()
                    volatility = returns.std()
                    if self._min_volatility <= volatility <= self._max_volatility:
                        filtered.append(pair)
        return filtered


class AgeFilter(PairListHandler):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._min_days_listed = config.get("min_days_listed", 1)

    def filter_pairlist(self, pairlist: list[str], market_data: dict[str, pd.DataFrame]) -> list[str]:
        filtered = []
        for pair in pairlist:
            if pair in market_data:
                df = market_data[pair]
                if len(df) >= self._min_days_listed * 288:
                    filtered.append(pair)
        return filtered


class StaticPairList(PairListHandler):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._pair_whitelist = config.get("pair_whitelist", [])

    def filter_pairlist(self, pairlist: list[str], market_data: dict[str, pd.DataFrame]) -> list[str]:
        if self._pair_whitelist:
            return [p for p in self._pair_whitelist if p in pairlist]
        return pairlist


class PairListManager:
    def __init__(self, config: list[dict[str, Any]]):
        self._handlers: list[PairListHandler] = []
        self._load_handlers(config)

    def _load_handlers(self, config: list[dict[str, Any]]) -> None:
        handler_map = {
            "VolumePairList": VolumePairList,
            "PriceFilter": PriceFilter,
            "SpreadFilter": SpreadFilter,
            "VolatilityFilter": VolatilityFilter,
            "AgeFilter": AgeFilter,
            "StaticPairList": StaticPairList,
        }
        
        for handler_config in config:
            method = handler_config.get("method")
            if method in handler_map:
                self._handlers.append(handler_map[method](handler_config))
                logger.info(f"Loaded pairlist handler: {method}")

    def get_valid_pairs(self, available_pairs: list[str], market_data: dict[str, pd.DataFrame]) -> list[str]:
        pairs = available_pairs
        for handler in self._handlers:
            pairs = handler.filter_pairlist(pairs, market_data)
            logger.info(f"After {handler.__class__.__name__}: {len(pairs)} pairs")
        return pairs
