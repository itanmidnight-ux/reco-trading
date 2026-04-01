from __future__ import annotations

from reco_trading.data.candle_builder import ohlcv_to_frame


class MarketStream:
    """Pulls market candles from exchange client and exposes normalized frames."""

    def __init__(self, client, symbol: str, history_limit: int = 300) -> None:
        self.client = client
        self.symbol = symbol
        self.history_limit = max(int(history_limit), 1)

    async def fetch_frame(self, timeframe: str):
        ohlcv = await self.client.fetch_ohlcv(self.symbol, timeframe=timeframe, limit=self.history_limit)
        return ohlcv_to_frame(ohlcv)
