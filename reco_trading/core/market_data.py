from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from reco_trading.infra.binance_client import BinanceClient


@dataclass(frozen=True, slots=True)
class MarketQuality:
    operable: bool
    reason: str
    spread_bps: float
    realized_volatility: float
    avg_volume: float
    gap_ratio: float


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
        frame = frame.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last').reset_index(drop=True)
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

    @staticmethod
    def _gap_ratio(frame: pd.DataFrame) -> float:
        if frame.empty or len(frame) < 2:
            return 0.0
        ts = frame['timestamp'].astype('int64')
        deltas = ts.diff().dropna()
        if deltas.empty:
            return 0.0
        expected = float(deltas.median())
        if expected <= 0.0:
            return 0.0
        abnormal = float((deltas > (1.8 * expected)).mean())
        return max(abnormal, 0.0)

    def assess_market_quality(
        self,
        frame: pd.DataFrame,
        *,
        spread_bps: float,
        max_spread_bps: float,
        max_volatility: float,
        min_avg_volume: float,
        max_gap_ratio: float,
    ) -> MarketQuality:
        close = frame['close'].astype(float)
        volume = frame['volume'].astype(float)
        returns = np.log(close / close.shift(1)).dropna()
        realized_vol = float(returns.tail(60).std() or 0.0)
        avg_volume = float(volume.tail(60).mean() or 0.0)
        gap_ratio = self._gap_ratio(frame.tail(300))

        if spread_bps > max_spread_bps:
            return MarketQuality(False, 'spread_anomalo', spread_bps, realized_vol, avg_volume, gap_ratio)
        if realized_vol > max_volatility:
            return MarketQuality(False, 'volatilidad_extrema', spread_bps, realized_vol, avg_volume, gap_ratio)
        if avg_volume < min_avg_volume:
            return MarketQuality(False, 'mercado_muerto_bajo_volumen', spread_bps, realized_vol, avg_volume, gap_ratio)
        if gap_ratio > max_gap_ratio:
            return MarketQuality(False, 'gaps_anomalos_ohlcv', spread_bps, realized_vol, avg_volume, gap_ratio)
        return MarketQuality(True, 'market_operable', spread_bps, realized_vol, avg_volume, gap_ratio)

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
