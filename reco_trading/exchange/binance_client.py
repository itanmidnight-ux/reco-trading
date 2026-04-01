from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

import ccxt


# ============================================================
# EXCHANGE RATE LIMIT CONFIGURATION
# All limits based on official exchange documentation
# ============================================================

EXCHANGE_RATE_LIMITS = {
    "binance": {
        "spot": {"rpm": 1200, "weight_per_minute": 6000, "orders_per_second": 10},
        "futures": {"rpm": 2400, "weight_per_minute": 2400, "orders_per_second": 10},
        "margin": {"rpm": 1200, "weight_per_minute": 6000, "orders_per_second": 5},
        "testnet": {"rpm": 1200, "weight_per_minute": 6000, "orders_per_second": 5},
    },
    "binanceus": {"spot": {"rpm": 600, "weight_per_minute": 6000, "orders_per_second": 5}},
    "kraken": {"spot": {"rpm": 900, "orders_per_second": 1}},
    "kucoin": {"spot": {"rpm": 600, "orders_per_second": 2}, "futures": {"rpm": 600, "orders_per_second": 5}},
    "bybit": {"spot": {"rpm": 600, "orders_per_second": 10}, "futures": {"rpm": 600, "orders_per_second": 10}},
    "okx": {"spot": {"rpm": 600, "orders_per_second": 10}, "futures": {"rpm": 600, "orders_per_second": 10}},
    "gate": {"spot": {"rpm": 900, "orders_per_second": 5}},
    "bitget": {"spot": {"rpm": 600, "orders_per_second": 5}},
    "huobi": {"spot": {"rpm": 600, "orders_per_second": 5}},
    "mexc": {"spot": {"rpm": 600, "orders_per_second": 5}},
    "coinbase": {"spot": {"rpm": 600, "orders_per_second": 10}},
    "bitstamp": {"spot": {"rpm": 600, "orders_per_second": 1}},
}

DEFAULT_RATE_LIMIT = {"rpm": 300, "orders_per_second": 1}


class BinanceClient:
    """
    Async wrapper for Binance via ccxt with retry, time sync, and robust error handling.
    Uses sync ccxt wrapped in asyncio.to_thread() (proven approach from backup).
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        trading_mode: str = "spot",
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.trading_mode = trading_mode
        self.time_offset_ms = 0
        self._time_sync_lock = asyncio.Lock()
        self._last_time_sync_monotonic = 0.0
        self._markets_loaded = False

        # Create exchange immediately in __init__ (proven approach from backup)
        self.exchange = ccxt.binance({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": trading_mode,
                "adjustForTimeDifference": True,
                "recvWindow": 60000,
                "timestamp": True,
            },
        })

        if testnet:
            self.exchange.set_sandbox_mode(True)
            self.logger.info("Testnet mode enabled")

    async def connect(self) -> bool:
        """Establish connection - loads markets and syncs time. Idempotent."""
        try:
            if not self.api_key or not self.api_secret:
                raise ValueError(
                    "API keys not configured. Please set BINANCE_API_KEY and BINANCE_API_SECRET in .env file."
                )

            # Load markets (required before any trading)
            if not self._markets_loaded:
                await asyncio.to_thread(self.exchange.load_markets)
                self._markets_loaded = True

            # Sync time with retry
            for attempt in range(3):
                try:
                    await self.sync_time()
                    break
                except Exception as e:
                    self.logger.warning(f"Time sync attempt {attempt+1} failed: {e}")
                    if attempt == 2:
                        self.logger.error("Time sync failed after 3 attempts, continuing anyway")
                    await asyncio.sleep(1)

            # Verify API keys work by fetching balance
            try:
                balance = await self.fetch_balance()
                usdt_balance = balance.get("USDT", {}).get("free", 0)
                self.logger.info(
                    f"Binance connected: testnet={self.testnet}, mode={self.trading_mode}, "
                    f"markets={len(self.exchange.markets)}, time_offset={self.time_offset_ms}ms, "
                    f"USDT balance={usdt_balance}"
                )
            except ccxt.AuthenticationError as e:
                self.logger.error(
                    f"API authentication failed. Your API keys may be invalid or expired.\n"
                    f"  - Testnet keys expire after 90 days of inactivity\n"
                    f"  - Generate new keys at: https://testnet.binance.vision/\n"
                    f"  - Error: {e}"
                )
                raise
            except Exception as e:
                self.logger.warning(f"Balance check failed (non-critical): {e}")

            return True

        except ccxt.AuthenticationError as e:
            self.logger.error(f"Binance authentication failed: {e}")
            raise
        except ccxt.ExchangeError as e:
            self.logger.error(f"Binance exchange error: {e}")
            raise
        except ValueError as e:
            self.logger.error(f"Configuration error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Binance connection failed: {e}")
            raise

    async def sync_exchange_time(self, reason: str = "manual") -> None:
        """Synchronize local/exchange time drift without crashing the bot."""
        async with self._time_sync_lock:
            try:
                self.logger.info("binance_time_sync_start reason=%s", reason)
                # Fetch server time and compute offset
                server_time = await asyncio.to_thread(self.exchange.fetch_time)
                local_time = int(time.time() * 1000)
                self.time_offset_ms = server_time - local_time
                
                # Patch ccxt's nonce method (used for signed request timestamps)
                # This is the most reliable way to fix timestamp drift
                offset = self.time_offset_ms
                original_nonce = self.exchange.nonce
                def patched_nonce():
                    return int(time.time() * 1000) + offset
                self.exchange.nonce = patched_nonce
                
                # Also set timeDifference for compatibility
                self.exchange.timeDifference = self.time_offset_ms
                self.exchange.options["timeDifference"] = self.time_offset_ms
                
                self._last_time_sync_monotonic = time.monotonic()
                self.logger.info("binance_time_sync_ok reason=%s offset_ms=%s", reason, self.time_offset_ms)
            except Exception as exc:
                self.logger.warning("binance_time_sync_failed reason=%s error=%s", reason, exc)

    async def sync_time(self) -> None:
        """Backward-compatible alias used by current bot startup."""
        await self.sync_exchange_time(reason="startup")

    async def periodic_time_resync(self, interval_seconds: float = 1800.0) -> None:
        """Periodically resync time if needed."""
        if (time.monotonic() - self._last_time_sync_monotonic) < max(float(interval_seconds), 1.0):
            return
        await self.sync_exchange_time(reason="periodic")

    async def load_markets(self) -> dict[str, Any]:
        """Load markets from exchange. Cached after first call."""
        if not self._markets_loaded:
            await asyncio.to_thread(self.exchange.load_markets)
            self._markets_loaded = True
        return self.exchange.markets

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "5m", limit: int = 300) -> list[list[float]]:
        """Fetch OHLCV candlestick data."""
        return await self._with_retry(
            self.exchange.fetch_ohlcv, symbol, timeframe, None, limit,
            operation="fetch_ohlcv"
        )

    async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        """Fetch ticker data for a symbol."""
        return await self._with_retry(
            self.exchange.fetch_ticker, symbol,
            operation="fetch_ticker"
        )

    async def fetch_order_book(self, symbol: str, limit: int = 20) -> dict[str, Any]:
        """Fetch order book."""
        return await self._with_retry(
            self.exchange.fetch_order_book, symbol, limit,
            operation="fetch_order_book"
        )

    async def fetch_balance(self) -> dict[str, Any]:
        """Fetch account balance."""
        return await self._with_retry(
            self.exchange.fetch_balance,
            operation="fetch_balance"
        )

    async def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        *,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a market order."""
        params: dict[str, Any] = {}
        if client_order_id:
            params["newClientOrderId"] = client_order_id
        return await self._with_retry(
            self.exchange.create_order, symbol, "market", side, amount, None, params,
            operation="create_market_order"
        )

    async def fetch_order(self, order_id: str, symbol: str) -> dict[str, Any]:
        """Fetch order by ID."""
        return await self._with_retry(
            self.exchange.fetch_order, order_id, symbol,
            operation="fetch_order"
        )

    async def fetch_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Fetch open orders."""
        return await self._with_retry(
            self.exchange.fetch_open_orders, symbol,
            operation="fetch_open_orders"
        )

    async def fetch_my_trades(self, symbol: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Fetch trade history."""
        return await self._with_retry(
            self.exchange.fetch_my_trades, symbol, None, limit,
            operation="fetch_my_trades"
        )

    async def cancel_order(self, order_id: str, symbol: str) -> dict[str, Any]:
        """Cancel an order."""
        return await self._with_retry(
            self.exchange.cancel_order, order_id, symbol,
            operation="cancel_order"
        )

    async def close(self) -> None:
        """Close exchange connection."""
        close_fn = getattr(self.exchange, "close", None)
        if callable(close_fn):
            try:
                await asyncio.to_thread(close_fn)
            except Exception:
                pass

    # ============================================================
    # RETRY & ERROR HANDLING
    # ============================================================

    async def _with_retry(self, fn: Callable[..., Any], *args: Any, retries: int = 3, operation: str = "unknown", **kwargs: Any) -> Any:
        """Execute exchange call with retry, timestamp resync, and exponential backoff."""
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
                        operation, attempt + 1, retries,
                    )
                    await self.sync_exchange_time(reason=f"timestamp_error:{operation}")
                    continue
                raise
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as exc:
                self.logger.warning(
                    "binance_retry op=%s attempt=%s/%s error=%s",
                    operation, attempt + 1, retries, exc,
                )
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(delay)
                delay *= 2

        raise RuntimeError(f"exchange_call_exhausted op={operation}")

    @staticmethod
    def _is_timestamp_error(exc: Exception) -> bool:
        message = str(exc)
        return "-1021" in message or "outside of the recvwindow" in message.lower()
