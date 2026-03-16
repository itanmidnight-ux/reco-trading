from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

import ccxt


class BinanceClient:
    """Async wrapper for Binance via ccxt with retry and time sync safety."""

    def __init__(self, api_key: str, api_secret: str, testnet: bool) -> None:
        self.logger = logging.getLogger(__name__)
        self.exchange = ccxt.binance(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "spot", "adjustForTimeDifference": True},
            }
        )
        self.time_offset_ms = 0
        if testnet:
            self.exchange.set_sandbox_mode(True)

    async def sync_time(self) -> None:
        server_ms = await self._call_with_retry(self.exchange.fetch_time, operation="sync_time", retries=5)
        local_ms = int(time.time() * 1000)
        self.time_offset_ms = int(server_ms) - local_ms
        self.exchange.options["timeDifference"] = self.time_offset_ms

    async def load_markets(self) -> dict[str, Any]:
        return await self._call_with_retry(self.exchange.load_markets, operation="load_markets", retries=5)

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
        return await self._call_with_retry(self.exchange.fetch_ohlcv, symbol, timeframe, None, limit, operation="fetch_ohlcv")

    async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        return await self._call_with_retry(self.exchange.fetch_ticker, symbol, operation="fetch_ticker")

    async def fetch_order_book(self, symbol: str, limit: int = 20) -> dict[str, Any]:
        return await self._call_with_retry(self.exchange.fetch_order_book, symbol, limit, operation="fetch_order_book")

    async def fetch_balance(self) -> dict[str, Any]:
        return await self._call_with_retry(self.exchange.fetch_balance, operation="fetch_balance")

    async def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        *,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if client_order_id:
            params["newClientOrderId"] = client_order_id
        return await self._call_with_retry(
            self.exchange.create_order,
            symbol,
            "market",
            side,
            amount,
            None,
            params,
            operation="create_market_order",
            retries=4,
        )

    async def close(self) -> None:
        await asyncio.to_thread(self.exchange.close)

    async def _call_with_retry(
        self,
        fn: Callable[..., Any],
        *args: Any,
        retries: int = 3,
        operation: str = "unknown",
        **kwargs: Any,
    ) -> Any:
        delay = 1.0
        for attempt in range(retries):
            try:
                return await asyncio.to_thread(fn, *args, **kwargs)
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as exc:
                self.logger.warning("binance_retry op=%s attempt=%s/%s error=%s", operation, attempt + 1, retries, exc)
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(delay)
                delay *= 2
