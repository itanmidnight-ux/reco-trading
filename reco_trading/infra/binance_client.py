from __future__ import annotations

import asyncio
import ccxt.async_support as ccxt


class BinanceClient:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False) -> None:
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
        })
        if testnet:
            self.exchange.set_sandbox_mode(True)

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500):
        return await self.exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)

    async def fetch_balance(self):
        return await self.exchange.fetch_balance()

    async def create_market_order(self, symbol: str, side: str, amount: float):
        return await self.exchange.create_order(symbol, 'market', side.lower(), amount)

    async def fetch_order(self, symbol: str, order_id: str):
        return await self.exchange.fetch_order(order_id, symbol)

    async def wait_for_fill(self, symbol: str, order_id: str, timeout: int = 30):
        for _ in range(timeout):
            order = await self.fetch_order(symbol, order_id)
            if order.get('status') == 'closed':
                return order
            await asyncio.sleep(1)
        return None

    async def close(self) -> None:
        await self.exchange.close()
