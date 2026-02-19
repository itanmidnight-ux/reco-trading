from __future__ import annotations

import pandas as pd

from reco_trading.infra.binance_client import BinanceClient


class MarketDataService:
    def __init__(self, client: BinanceClient, symbol: str, timeframe: str) -> None:
        self.client = client
        self.symbol = symbol
        self.timeframe = timeframe

    async def latest_ohlcv(self, limit: int = 500) -> pd.DataFrame:
        rows = await self.client.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
        frame = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        frame['timestamp'] = pd.to_datetime(frame['timestamp'], unit='ms', utc=True)
        return frame
