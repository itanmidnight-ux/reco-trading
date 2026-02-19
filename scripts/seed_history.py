from __future__ import annotations

import asyncio

from trading_system.app.config.settings import Settings
from trading_system.app.core.rate_limiter import BinanceRateLimitController
from trading_system.app.database.repository import Repository
from trading_system.app.services.market_data.binance_client import BinanceClient


async def main() -> None:
    s = Settings()
    repo = Repository(s.postgres_dsn)
    limiter = BinanceRateLimitController(s.binance_max_weight, s.order_rate_per_sec)
    client = BinanceClient(s, limiter)

    await repo.create_schema()
    klines = await client.get_klines(s.symbol, '1m', 200)
    for k in klines:
        payload = {
            'k': {
                'x': True,
                'T': k[6],
                'o': k[1],
                'h': k[2],
                'l': k[3],
                'c': k[4],
                'v': k[5],
            }
        }
        await repo.save_candle(s.symbol, '1m', payload)


if __name__ == '__main__':
    asyncio.run(main())
