from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp
import ccxt.async_support as ccxt
from ccxt.base.errors import DDoSProtection, ExchangeError, NetworkError, RateLimitExceeded

from reco_trading.core.rate_limit_controller import AdaptiveRateLimitController


logger = logging.getLogger(__name__)


class BinanceClient:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True, confirm_mainnet: bool = False) -> None:
        if not api_key.strip() or not api_secret.strip():
            raise ValueError('api_key y api_secret de Binance son obligatorias y no pueden estar vacías.')
        if not testnet and not confirm_mainnet:
            raise ValueError('Mainnet requiere confirm_mainnet=true explícito por seguridad institucional.')

        self.rest_base, self.ws_base = self._resolve_endpoints(testnet)
        self._symbol = 'BTC/USDT'
        self._log_selected_endpoint(testnet=testnet)
        self.exchange = ccxt.binance(
            {
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot', 'fetchCurrencies': False},
            }
        )
        self.exchange.urls['api']['public'] = self.rest_base + '/v3'
        self.exchange.urls['api']['private'] = self.rest_base + '/v3'
        self._rate_limiter = AdaptiveRateLimitController(max_calls=8, period_seconds=1.0)
        self._ws_backoff_seconds = 1.0
        self._markets_loaded = False
        self._symbol_rules_cache: dict[str, dict[str, float | None]] = {}
        if testnet:
            self.exchange.set_sandbox_mode(True)

    async def initialize(self) -> None:
        if self._markets_loaded:
            return
        await self.exchange.load_markets()
        self._markets_loaded = True

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
        await self.initialize()
        return await self._retry(self.exchange.fetch_ohlcv, symbol=symbol, timeframe=timeframe, limit=limit)

    async def fetch_order_book(self, symbol: str, limit: int = 20) -> Any:
        await self.initialize()
        return await self._retry(self.exchange.fetch_order_book, symbol=symbol, limit=limit)

    async def fetch_balance(self) -> Any:
        await self.initialize()
        return await self._retry(self.exchange.fetch_balance)

    async def ping(self) -> Any:
        try:
            await self.initialize()
            ticker = await self.exchange.fetch_ticker(self._symbol)
            if not ticker or ticker.get('last') is None:
                raise RuntimeError('Ticker inválido en ping.')
            return True
        except Exception as e:
            raise RuntimeError(f'Binance ping failed: {e}') from e

    async def fetch_ticker(self, symbol: str) -> Any:
        await self.initialize()
        return await self._retry(self.exchange.fetch_ticker, symbol=symbol)

    async def get_symbol_rules(self, symbol: str) -> dict[str, float | None]:
        await self.initialize()
        cached = self._symbol_rules_cache.get(symbol)
        if cached is not None:
            return cached

        market = self.exchange.market(symbol)
        limits = market.get('limits') or {}
        amount_limits = limits.get('amount') or {}
        cost_limits = limits.get('cost') or {}
        rules = {
            'min_qty': float(amount_limits.get('min') or 0.0),
            'max_qty': float(amount_limits.get('max') or 0.0) or None,
            'min_notional': float(cost_limits.get('min') or 0.0),
        }
        self._symbol_rules_cache[symbol] = rules
        return rules

    async def sanitize_order_quantity(self, symbol: str, amount: float, reference_price: float) -> float:
        await self.initialize()
        if amount <= 0:
            raise ValueError('Cantidad inválida: debe ser mayor a cero')

        normalized = float(self.exchange.amount_to_precision(symbol, amount))
        rules = await self.get_symbol_rules(symbol)

        min_qty = float(rules.get('min_qty') or 0.0)
        max_qty = rules.get('max_qty')
        min_notional = float(rules.get('min_notional') or 0.0)

        if normalized < min_qty:
            raise ValueError(f'Cantidad {normalized} por debajo del mínimo permitido {min_qty} para {symbol}')
        if max_qty is not None and normalized > float(max_qty):
            raise ValueError(f'Cantidad {normalized} supera el máximo permitido {max_qty} para {symbol}')
        if reference_price <= 0:
            raise ValueError('Precio de referencia inválido para validar notional')

        notional = normalized * reference_price
        if min_notional > 0.0 and notional < min_notional:
            raise ValueError(
                f'Notional {notional:.8f} por debajo del mínimo {min_notional:.8f} en {symbol}'
            )
        return normalized

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
        await self.initialize()
        if side.upper() == 'BUY':
            return await self._retry(self.exchange.create_market_buy_order, symbol, amount)
        if side.upper() == 'SELL':
            return await self._retry(self.exchange.create_market_sell_order, symbol, amount)
        raise ValueError(f'Lado de orden inválido: {side}')

    async def fetch_order(self, symbol: str, order_id: str) -> Any:
        await self.initialize()
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
