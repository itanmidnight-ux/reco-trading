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
                "recvWindow": 10000,
                "options": {"defaultType": "spot", "adjustForTimeDifference": True, "recvWindow": 10000},
            }
        )
        self.time_offset_ms = 0
        self._time_sync_lock = asyncio.Lock()
        self._last_time_sync_monotonic = 0.0
        if testnet:
            self.exchange.set_sandbox_mode(True)

    async def sync_exchange_time(self, reason: str = "manual") -> None:
        """Synchronize local/exchange time drift without crashing the bot."""
        async with self._time_sync_lock:
            try:
                self.logger.info("binance_time_sync_start reason=%s", reason)
                await asyncio.to_thread(self.exchange.load_time_difference)
                self.time_offset_ms = int(getattr(self.exchange, "timeDifference", 0) or 0)
                self.exchange.options["timeDifference"] = self.time_offset_ms
                self._last_time_sync_monotonic = time.monotonic()
                self.logger.info("binance_time_sync_ok reason=%s offset_ms=%s", reason, self.time_offset_ms)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("binance_time_sync_failed reason=%s error=%s", reason, exc)

    async def sync_time(self) -> None:
        # Backward-compatible alias used by current bot startup.
        await self.sync_exchange_time(reason="startup")

    async def periodic_time_resync(self, interval_seconds: float = 1800.0) -> None:
        if (time.monotonic() - self._last_time_sync_monotonic) < max(float(interval_seconds), 1.0):
            return
        await self.sync_exchange_time(reason="periodic")

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
        return await self.safe_exchange_call(self.exchange.create_order, symbol, "market", side, amount, None, params, operation="create_market_order")

    async def close(self) -> None:
        close_fn = getattr(self.exchange, "close", None)
        if callable(close_fn):
            await asyncio.to_thread(close_fn)

    async def safe_exchange_call(
        self,
        fn: Callable[..., Any],
        *args: Any,
        operation: str = "unknown",
        retries: int = 3,
        **kwargs: Any,
    ) -> Any:
        """Execute exchange request with retry + timestamp auto-resync and single retry."""
        delay = 1.0
        timestamp_resynced = False

        for attempt in range(retries):
            try:
                return await asyncio.to_thread(fn, *args, **kwargs)
            except (ccxt.InvalidNonce, ccxt.ExchangeError) as exc:
                if self._is_timestamp_error(exc) and not timestamp_resynced:
                    timestamp_resynced = True
                    self.logger.warning(
                        "binance_timestamp_drift_detected op=%s attempt=%s/%s action=resync_and_retry",
                        operation,
                        attempt + 1,
                        retries,
                    )
                    await self.sync_exchange_time(reason=f"timestamp_error:{operation}")
                    self.logger.warning("binance_timestamp_retry op=%s", operation)
                    continue
                raise
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as exc:
                self.logger.warning("binance_retry op=%s attempt=%s/%s error=%s", operation, attempt + 1, retries, exc)
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(delay)
                delay *= 2

        raise RuntimeError(f"exchange_call_exhausted op={operation}")

    async def _call_with_retry(
        self,
        fn: Callable[..., Any],
        *args: Any,
        retries: int = 3,
        operation: str = "unknown",
        **kwargs: Any,
    ) -> Any:
        return await self.safe_exchange_call(fn, *args, retries=retries, operation=operation, **kwargs)

    @staticmethod
    def _is_timestamp_error(exc: Exception) -> bool:
        message = str(exc)
        return "-1021" in message or "outside of the recvwindow" in message.lower()
