from __future__ import annotations

import logging

import pandas as pd

from reco_trading.data.candle_builder import ohlcv_to_frame
from reco_trading.exchange.binance_client import BinanceClient


class MarketStream:
    """Market data polling service."""

    def __init__(self, client: BinanceClient, symbol: str, limit: int) -> None:
        self.client = client
        self.symbol = symbol
        self.limit = limit
        self.logger = logging.getLogger(__name__)

    async def fetch_frame(self, timeframe: str) -> pd.DataFrame:
        ohlcv = await self.client.fetch_ohlcv(self.symbol, timeframe, self.limit)
        return ohlcv_to_frame(ohlcv)
