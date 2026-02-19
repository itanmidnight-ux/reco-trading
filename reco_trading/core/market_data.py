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

    async def live_preview(self, symbol_rest: str):
        async for event in self.client.stream_klines(symbol_rest, self.timeframe):
            kline = event.get('k', {})
            yield {
                'close_time': kline.get('T'),
                'open': float(kline.get('o', 0.0)),
                'high': float(kline.get('h', 0.0)),
                'low': float(kline.get('l', 0.0)),
                'close': float(kline.get('c', 0.0)),
                'volume': float(kline.get('v', 0.0)),
            }
