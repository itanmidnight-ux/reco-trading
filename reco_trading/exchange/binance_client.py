from __future__ import annotations

import asyncio
import logging
from typing import Any

import ccxt


class BinanceClient:
    """Async wrapper for Binance via ccxt."""

    def __init__(self, api_key: str, api_secret: str, testnet: bool) -> None:
        self.logger = logging.getLogger(__name__)
        self.exchange = ccxt.binance(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            }
        )
        if testnet:
            self.exchange.set_sandbox_mode(True)

    async def load_markets(self) -> dict[str, Any]:
        return await asyncio.to_thread(self.exchange.load_markets)

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
        return await asyncio.to_thread(self.exchange.fetch_ohlcv, symbol, timeframe, None, limit)

    async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        return await asyncio.to_thread(self.exchange.fetch_ticker, symbol)

    async def fetch_balance(self) -> dict[str, Any]:
        return await asyncio.to_thread(self.exchange.fetch_balance)

    async def create_market_order(self, symbol: str, side: str, amount: float) -> dict[str, Any]:
        return await asyncio.to_thread(self.exchange.create_order, symbol, "market", side, amount)

    async def close(self) -> None:
        await asyncio.to_thread(self.exchange.close)
