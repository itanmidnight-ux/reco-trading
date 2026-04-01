from __future__ import annotations


class DataProvider:
    """Small compatibility wrapper used by pairlist filters."""

    def __init__(self, exchange) -> None:
        self.exchange = exchange

    def ohlcv(self, pair: str, timeframe: str = "1h", limit: int = 100):
        fetch = getattr(self.exchange, "fetch_ohlcv", None)
        if not callable(fetch):
            return []
        return fetch(pair, timeframe=timeframe, limit=limit)
