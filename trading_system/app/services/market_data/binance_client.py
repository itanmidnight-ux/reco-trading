from __future__ import annotations

import time
from typing import Any

import aiohttp

from trading_system.app.config.settings import Settings
from trading_system.app.core.rate_limiter import BinanceRateLimitController, exponential_backoff
from trading_system.app.core.security import sign_params


class BinanceClient:
    def __init__(self, settings: Settings, limiter: BinanceRateLimitController) -> None:
        self.settings = settings
        self.limiter = limiter
        self.rest_base = 'https://testnet.binance.vision' if settings.binance_testnet else 'https://api.binance.com'
        self.ws_base = 'wss://testnet.binance.vision/stream?streams=' if settings.binance_testnet else 'wss://stream.binance.com:9443/stream?streams='

    async def request(self, method: str, path: str, *, params: dict[str, Any] | None = None, private: bool = False, weight: int = 1) -> dict[str, Any]:
        params = params or {}
        headers = {}
        if private:
            params['timestamp'] = int(time.time() * 1000)
            params['recvWindow'] = 5000
            params['signature'] = sign_params(self.settings.binance_api_secret, params)
            headers['X-MBX-APIKEY'] = self.settings.binance_api_key

        for attempt in range(8):
            await self.limiter.reserve_weight(weight)
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.request(method, f'{self.rest_base}{path}', params=params, headers=headers) as resp:
                    header_weight = resp.headers.get('X-MBX-USED-WEIGHT-1M')
                    if header_weight and header_weight.isdigit():
                        self.limiter.apply_header_usage(int(header_weight))
                    if resp.status == 429:
                        await exponential_backoff(attempt)
                        continue
                    if resp.status == 418:
                        self.limiter.trigger_418_cooldown(120)
                        await exponential_backoff(attempt + 2)
                        continue
                    resp.raise_for_status()
                    return await resp.json()
        raise RuntimeError(f'No se pudo completar {method} {path} por límites/API')

    async def get_klines(self, symbol: str, interval: str = '1m', limit: int = 1000) -> list[list[Any]]:
        return await self.request('GET', '/api/v3/klines', params={'symbol': symbol, 'interval': interval, 'limit': limit}, weight=2)

    async def get_account(self) -> dict[str, Any]:
        return await self.request('GET', '/api/v3/account', private=True, weight=20)

    async def create_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        await self.limiter.reserve_order_slot()
        return await self.request('POST', '/api/v3/order', params=payload, private=True, weight=1)

    async def cancel_order(self, symbol: str, order_id: int) -> dict[str, Any]:
        await self.limiter.reserve_order_slot()
        return await self.request('DELETE', '/api/v3/order', params={'symbol': symbol, 'orderId': order_id}, private=True, weight=1)

    async def open_orders(self, symbol: str) -> list[dict[str, Any]]:
        return await self.request('GET', '/api/v3/openOrders', params={'symbol': symbol}, private=True, weight=3)

    async def order_status(self, symbol: str, order_id: int) -> dict[str, Any]:
        return await self.request('GET', '/api/v3/order', params={'symbol': symbol, 'orderId': order_id}, private=True, weight=2)
