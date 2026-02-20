from __future__ import annotations

import asyncio
import json

import aiohttp
import ccxt.async_support as ccxt
from ccxt.base.errors import DDoSProtection, ExchangeError, NetworkError, RateLimitExceeded

from reco_trading.core.rate_limit_controller import AdaptiveRateLimitController


class BinanceClient:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False) -> None:
        self.exchange = ccxt.binance(
            {
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'},
            }
        )
        self._rate_limiter = AdaptiveRateLimitController(max_calls=8, period_seconds=1.0)
        self._ws_backoff_seconds = 1.0
        if testnet:
            self.exchange.set_sandbox_mode(True)

    async def _retry(self, fn, *args, retries: int = 7, **kwargs):
        for attempt in range(1, retries + 1):
            try:
                await self._rate_limiter.acquire()
                return await fn(*args, **kwargs)
            except (RateLimitExceeded, NetworkError, DDoSProtection):
                wait = min(2**attempt, 30)
                await asyncio.sleep(wait)
            except ExchangeError:
                if attempt == retries:
                    raise
                await asyncio.sleep(min(attempt, 5))

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500):
        return await self._retry(self.exchange.fetch_ohlcv, symbol=symbol, timeframe=timeframe, limit=limit)

    async def fetch_balance(self):
        return await self._retry(self.exchange.fetch_balance)

    async def create_market_order(self, symbol: str, side: str, amount: float):
        return await self._retry(self.exchange.create_order, symbol, 'market', side.lower(), amount)

    async def fetch_order(self, symbol: str, order_id: str):
        return await self._retry(self.exchange.fetch_order, order_id, symbol)

    async def wait_for_fill(self, symbol: str, order_id: str, timeout: int = 45):
        for _ in range(timeout):
            order = await self.fetch_order(symbol, order_id)
            if order.get('status') in {'closed', 'filled'}:
                return order
            await asyncio.sleep(1)
        return None

    async def stream_klines(self, symbol_rest: str, interval: str):
        url = f'wss://stream.binance.com:9443/ws/{symbol_rest.lower()}@kline_{interval}'
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(url, heartbeat=20) as ws:
                        self._ws_backoff_seconds = 1.0
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                payload = json.loads(msg.data)
                                if payload.get('k', {}).get('x'):
                                    yield payload
                            elif msg.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR}:
                                break
            except Exception:
                await asyncio.sleep(self._ws_backoff_seconds)
                self._ws_backoff_seconds = min(self._ws_backoff_seconds * 2.0, 30.0)

    async def close(self) -> None:
        await self.exchange.close()
