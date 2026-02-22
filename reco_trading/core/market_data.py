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
        if frame.empty:
            raise ValueError('OHLCV vacío desde exchange')
        frame = frame.replace([float('inf'), float('-inf')], pd.NA).dropna()
        frame['timestamp'] = pd.to_datetime(frame['timestamp'], unit='ms', utc=True)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        frame[numeric_cols] = frame[numeric_cols].astype(float)
        if (frame[numeric_cols] < 0).any().any():
            raise ValueError('OHLCV corrupto: valores negativos detectados')
        if (frame['close'] <= 0).any():
            raise ValueError('Invalid price received from Binance')
        return frame

    async def latest_order_book(self, limit: int = 20) -> dict:
        book = await self.client.fetch_order_book(self.symbol, limit=limit)
        if not book or 'bids' not in book or 'asks' not in book:
            raise ValueError('Order book no disponible')
        return book

    async def latest_spread_bps(self, limit: int = 10) -> float:
        book = await self.latest_order_book(limit=limit)
        bids = book.get('bids') or []
        asks = book.get('asks') or []
        if not bids or not asks:
            return 0.0
        bid = float(bids[0][0])
        ask = float(asks[0][0])
        if bid <= 0.0 or ask <= 0.0 or ask < bid:
            return 0.0
        return ((ask - bid) / bid) * 10_000.0

    async def live_preview(self, symbol_rest: str):
        async for event in self.client.stream_klines(symbol_rest, self.timeframe):
            kline = event.get('k', {})
            close_price = float(kline.get('c', 0.0))
            if close_price <= 0:
                raise ValueError('Invalid price received from Binance')
            yield {
                'close_time': kline.get('T'),
                'open': float(kline.get('o', 0.0)),
                'high': float(kline.get('h', 0.0)),
                'low': float(kline.get('l', 0.0)),
                'close': close_price,
                'volume': float(kline.get('v', 0.0)),
            }
