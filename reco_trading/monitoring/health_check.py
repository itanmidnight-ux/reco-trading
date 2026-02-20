from __future__ import annotations

from typing import Any

from reco_trading.hft.safety import HFTSafetyMonitor
from reco_trading.infra.binance_client import BinanceClient


class HealthCheck:
    async def run(self, client: BinanceClient, symbol: str, timeframe: str) -> dict:
        rows = await client.fetch_ohlcv(symbol, timeframe, limit=5)
        return {'ok': len(rows) >= 5}

    def hft_safety(self, safety_monitor: HFTSafetyMonitor) -> dict[str, Any]:
        return {'ok': True, 'hft_safety': safety_monitor.health_snapshot()}
