from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp
import ccxt.async_support as ccxt
from ccxt.base.errors import DDoSProtection, ExchangeError, NetworkError, RateLimitExceeded

from reco_trading.core.rate_limit_controller import AdaptiveRateLimitController


logger = logging.getLogger(__name__)


class BinanceClient:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True, confirm_mainnet: bool = False) -> None:
        if not testnet and not confirm_mainnet:
            raise ValueError('Mainnet requiere confirm_mainnet=true explícito por seguridad institucional.')

        self.rest_base, self.ws_base = self._resolve_endpoints(testnet)
        self._log_selected_endpoint(testnet=testnet)
        self.exchange = ccxt.binance(
            {
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'},
            }
        )
        self.exchange.urls['api']['public'] = self.rest_base + '/v3'
        self.exchange.urls['api']['private'] = self.rest_base + '/v3'
        self._rate_limiter = AdaptiveRateLimitController(max_calls=8, period_seconds=1.0)
        self._ws_backoff_seconds = 1.0
        if testnet:
            self.exchange.set_sandbox_mode(True)

    @staticmethod
    def _resolve_endpoints(testnet: bool) -> tuple[str, str]:
        if testnet:
            return ('https://testnet.binance.vision/api', 'wss://stream.testnet.binance.vision')
        return ('https://api.binance.com/api', 'wss://stream.binance.com:9443')

    def _log_selected_endpoint(self, *, testnet: bool) -> None:
        if not self.rest_base.endswith('/api'):
            logger.warning('BinanceClient endpoint inesperado: rest_base=%s', self.rest_base)
        logger.info(
            'BinanceClient inicializado en %s | rest_base=%s | ws_base=%s',
            'testnet' if testnet else 'mainnet',
            self.rest_base,
            self.ws_base,
        )

    async def _retry(self, fn: Callable[..., Awaitable[Any]], *args: Any, retries: int = 7, **kwargs: Any) -> Any:
        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                await self._rate_limiter.acquire()
                return await fn(*args, **kwargs)
            except (RateLimitExceeded, NetworkError, DDoSProtection) as exc:
                last_exc = exc
                wait = min(2**attempt, 30)
                await asyncio.sleep(wait)
            except ExchangeError as exc:
                last_exc = exc
                if attempt == retries:
                    raise
                await asyncio.sleep(min(attempt, 5))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError('Retry loop finalizó sin resultado')

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> Any:
        return await self._retry(self.exchange.fetch_ohlcv, symbol=symbol, timeframe=timeframe, limit=limit)

    async def fetch_order_book(self, symbol: str, limit: int = 20) -> Any:
        return await self._retry(self.exchange.fetch_order_book, symbol=symbol, limit=limit)

    async def fetch_balance(self) -> Any:
        return await self._retry(self.exchange.fetch_balance)

    async def ping(self) -> Any:
        return await self._retry(self.exchange.fetch_status)

    async def fetch_ticker(self, symbol: str) -> Any:
        return await self._retry(self.exchange.fetch_ticker, symbol=symbol)

    async def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        *,
        firewall_checked: bool = False,
    ) -> Any:
        if not firewall_checked:
            raise PermissionError('create_market_order requiere validación previa del ExecutionFirewall')
        return await self._retry(self.exchange.create_order, symbol, 'market', side.lower(), amount)

    async def fetch_order(self, symbol: str, order_id: str) -> Any:
        return await self._retry(self.exchange.fetch_order, order_id, symbol)

    async def wait_for_fill(self, symbol: str, order_id: str, timeout: int = 45) -> Any:
        for _ in range(timeout):
            order = await self.fetch_order(symbol, order_id)
            if order.get('status') in {'closed', 'filled'}:
                return order
            await asyncio.sleep(1)
        return None

    async def stream_klines(self, symbol_rest: str, interval: str):
        url = f'{self.ws_base}/ws/{symbol_rest.lower()}@kline_{interval}'
        while True:
            try:
                timeout = aiohttp.ClientTimeout(total=None, connect=10, sock_read=30)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.ws_connect(url, heartbeat=20) as ws:
                        self._ws_backoff_seconds = 1.0
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    payload = json.loads(msg.data)
                                except json.JSONDecodeError:
                                    continue
                                if payload.get('k', {}).get('x'):
                                    yield payload
                            elif msg.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR}:
                                break
            except Exception:
                await asyncio.sleep(self._ws_backoff_seconds)
                self._ws_backoff_seconds = min(self._ws_backoff_seconds * 2.0, 30.0)

    async def close(self) -> None:
        await self.exchange.close()
