from __future__ import annotations

import pandas as pd

from bot.exchange.binance_client import BinanceClient


class MarketDataService:
    def __init__(self, client: BinanceClient) -> None:
        self.client = client

    async def get_candles(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        ohlcv = await self.client.fetch_ohlcv(symbol, timeframe, limit=limit)
        frame = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        frame['timestamp'] = pd.to_datetime(frame['timestamp'], unit='ms', utc=True)
        return frame


__all__ = ['MarketDataService']
