from __future__ import annotations

import asyncio
import time
from typing import Any

import ccxt.async_support as ccxt

from bot.config import BotConfig
from bot.utils.logger import logger


class BinanceClient:
    def __init__(self, config: BotConfig) -> None:
        self.config = config
        options = {
            'defaultType': 'spot',
            'adjustForTimeDifference': True,
            'recvWindow': config.recv_window,
        }
        self.exchange = ccxt.binance(
            {
                'apiKey': config.api_key,
                'secret': config.api_secret,
                'enableRateLimit': True,
                'timeout': config.request_timeout_ms,
                'options': options,
            }
        )
        if config.testnet:
            self.exchange.set_sandbox_mode(True)
        self._time_offset_ms = 0

    async def initialize(self) -> None:
        await self.sync_time()
        await self.exchange.load_markets()
        logger.info('Binance client initialized for {}', 'testnet' if self.config.testnet else 'mainnet')

    async def close(self) -> None:
        await self.exchange.close()

    async def sync_time(self) -> None:
        server_time = await self._with_retry(self.exchange.fetch_time)
        local_time = int(time.time() * 1000)
        self._time_offset_ms = server_time - local_time
        logger.info('Synced clock offset={}ms', self._time_offset_ms)

    def nonce(self) -> int:
        return int(time.time() * 1000 + self._time_offset_ms)

    async def _with_retry(self, fn, *args, **kwargs):
        delay = 0.5
        for attempt in range(1, self.config.retries + 1):
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                if attempt == self.config.retries:
                    raise
                logger.warning('Retry {}/{} after error: {}', attempt, self.config.retries, exc)
                await asyncio.sleep(delay)
                delay *= 2
                if 'timestamp' in str(exc).lower():
                    await self.sync_time()

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> list[list[float]]:
        return await self._with_retry(self.exchange.fetch_ohlcv, symbol, timeframe=timeframe, limit=limit)

    async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        return await self._with_retry(self.exchange.fetch_ticker, symbol)

    async def fetch_balance(self) -> dict[str, Any]:
        return await self._with_retry(self.exchange.fetch_balance)

    async def fetch_my_trades(self, symbol: str, since: int | None = None, limit: int = 100) -> list[dict[str, Any]]:
        return await self._with_retry(self.exchange.fetch_my_trades, symbol, since=since, limit=limit)

    async def create_order(self, symbol: str, side: str, amount: float, price: float | None = None) -> dict[str, Any]:
        params = {'recvWindow': self.config.recv_window, 'timestamp': self.nonce()}
        order_type = 'limit' if price else 'market'
        return await self._with_retry(
            self.exchange.create_order,
            symbol,
            order_type,
            side,
            amount,
            price,
            params,
        )

    async def fetch_market(self, symbol: str) -> dict[str, Any]:
        market = self.exchange.market(symbol)
        if not market:
            raise ValueError(f'Market not found: {symbol}')
        return market

    async def reconcile_order(self, symbol: str, order_id: str) -> dict[str, Any] | None:
        trades = await self.fetch_my_trades(symbol, limit=200)
        for trade in trades:
            if str(trade.get('order')) == str(order_id):
                return trade
        return None


__all__ = ['BinanceClient']
