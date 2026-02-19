from __future__ import annotations

from reco_trading.infra.binance_client import BinanceClient


class HealthCheck:
    async def run(self, client: BinanceClient, symbol: str, timeframe: str) -> dict:
        rows = await client.fetch_ohlcv(symbol, timeframe, limit=5)
        return {'ok': len(rows) >= 5}
