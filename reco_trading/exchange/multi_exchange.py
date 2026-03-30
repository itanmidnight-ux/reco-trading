from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import ccxt

logger = logging.getLogger(__name__)


EXCHANGE_FEATURES = {
    "binance": {
        "spot": True,
        "margin": True,
        "futures": True,
        "leverage": True,
        "has": {"fetchTickers": True, "fetchOHLCV": True, "createOrder": True},
    },
    "coinbase": {
        "spot": True,
        "margin": False,
        "futures": True,
        "leverage": False,
        "has": {"fetchTickers": True, "fetchOHLCV": True, "createOrder": True},
    },
    "kraken": {
        "spot": True,
        "margin": True,
        "futures": True,
        "leverage": True,
        "has": {"fetchTickers": True, "fetchOHLCV": True, "createOrder": True},
    },
    "kucoin": {
        "spot": True,
        "margin": True,
        "futures": True,
        "leverage": True,
        "has": {"fetchTickers": True, "fetchOHLCV": True, "createOrder": True},
    },
    "bybit": {
        "spot": True,
        "margin": True,
        "futures": True,
        "leverage": True,
        "has": {"fetchTickers": True, "fetchOHLCV": True, "createOrder": True},
    },
    "okx": {
        "spot": True,
        "margin": True,
        "futures": True,
        "leverage": True,
        "has": {"fetchTickers": True, "fetchOHLCV": True, "createOrder": True},
    },
    "gate": {
        "spot": True,
        "margin": True,
        "futures": True,
        "leverage": True,
        "has": {"fetchTickers": True, "fetchOHLCV": True, "createOrder": True},
    },
    "bitget": {
        "spot": True,
        "margin": True,
        "futures": True,
        "leverage": True,
        "has": {"fetchTickers": True, "fetchOHLCV": True, "createOrder": True},
    },
}


@dataclass
class ExchangeInfo:
    name: str
    exchange_id: str
    enabled: bool = True
    testnet: bool = True
    trading_mode: str = "spot"
    leverage: int = 1
    spot_enabled: bool = True
    margin_enabled: bool = False
    futures_enabled: bool = False


@dataclass
class ExchangeStatus:
    exchange_id: str
    connected: bool = False
    last_sync: datetime | None = None
    latency_ms: float = 0.0
    error_count: int = 0
    total_calls: int = 0


class MultiExchangeManager:
    """
    Full Multi-Exchange Manager supporting Spot, Margin, and Futures trading.
    """

    SUPPORTED_EXCHANGES = list(EXCHANGE_FEATURES.keys())

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.exchanges: dict[str, Any] = {}
        self.exchange_info: dict[str, ExchangeInfo] = {}
        self.exchange_status: dict[str, ExchangeStatus] = {}
        self.primary_exchange: str = "binance"
        self._default_exchange: str = "binance"

    def add_exchange(
        self,
        exchange_id: str,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        trading_mode: str = "spot",
        leverage: int = 1,
    ) -> bool:
        if exchange_id.lower() not in self.SUPPORTED_EXCHANGES:
            self.logger.error(f"Unsupported exchange: {exchange_id}")
            return False

        try:
            exchange_class = getattr(ccxt, exchange_id.lower())
            exchange = self._create_exchange_instance(
                exchange_class, api_key, api_secret, testnet, trading_mode
            )

            self.exchanges[exchange_id.lower()] = exchange
            self.exchange_info[exchange_id.lower()] = ExchangeInfo(
                name=EXCHANGE_FEATURES[exchange_id.lower()].get("name", exchange_id),
                exchange_id=exchange_id.lower(),
                testnet=testnet,
                trading_mode=trading_mode,
                leverage=leverage,
                spot_enabled=EXCHANGE_FEATURES[exchange_id.lower()].get("spot", False),
                margin_enabled=EXCHANGE_FEATURES[exchange_id.lower()].get("margin", False),
                futures_enabled=EXCHANGE_FEATURES[exchange_id.lower()].get("futures", False),
            )
            self.exchange_status[exchange_id.lower()] = ExchangeStatus(exchange_id=exchange_id.lower())

            self.logger.info(f"Added exchange: {exchange_id} (mode: {trading_mode}, leverage: {leverage}x)")
            return True

        except Exception as exc:
            self.logger.error(f"Failed to add exchange {exchange_id}: {exc}")
            return False

    def _create_exchange_instance(
        self,
        exchange_class: type,
        api_key: str,
        api_secret: str,
        testnet: bool,
        trading_mode: str,
    ) -> Any:
        config = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }

        testnet_urls = {
            "binance": "https://testnet.binance.vision/api",
            "kucoin": "https://api-sandbox.kucoin.com",
            "bybit": "https://api-testnet.bybit.com",
            "okx": "https://www.okx.com",
            "gate": "https://api-testnet.gateio.ws",
            "bitget": "https://api-testnet.bitget.com",
        }

        if testnet and testnet_urls.get(exchange_class.id):
            config["urls"] = {"api": testnet_urls[exchange_class.id]}

        if trading_mode == "futures":
            config["options"]["defaultType"] = "future"
        elif trading_mode == "margin":
            config["options"]["defaultType"] = "margin"

        return exchange_class(config)

    def set_primary(self, exchange_id: str) -> bool:
        if exchange_id.lower() in self.exchanges:
            self.primary_exchange = exchange_id.lower()
            self.logger.info(f"Primary exchange set to: {exchange_id}")
            return True
        return False

    def get_primary(self) -> Any:
        return self.exchanges.get(self.primary_exchange)

    def get_exchange(self, exchange_id: str | None = None) -> Any:
        if exchange_id:
            return self.exchanges.get(exchange_id.lower())
        return self.get_primary()

    def is_exchange_available(self, exchange_id: str) -> bool:
        return exchange_id.lower() in self.exchanges

    def get_exchange_features(self, exchange_id: str) -> dict | None:
        return EXCHANGE_FEATURES.get(exchange_id.lower())

    async def test_connection(self, exchange_id: str) -> bool:
        exchange = self.get_exchange(exchange_id)
        if not exchange:
            return False

        try:
            start = asyncio.get_event_loop().time()
            await asyncio.wait_for(asyncio.to_thread(exchange.fetch_time), timeout=10.0)
            latency = (asyncio.get_event_loop().time() - start) * 1000

            self.exchange_status[exchange_id.lower()].connected = True
            self.exchange_status[exchange_id.lower()].latency_ms = latency
            self.exchange_status[exchange_id.lower()].last_sync = datetime.now(timezone.utc)

            self.logger.info(f"Exchange {exchange_id} connected (latency: {latency:.2f}ms)")
            return True
        except Exception as exc:
            self.logger.error(f"Exchange {exchange_id} connection failed: {exc}")
            self.exchange_status[exchange_id.lower()].connected = False
            return False

    async def fetch_tickers(self, symbols: list[str] | None = None, exchange_id: str | None = None) -> dict[str, Any]:
        exchange = self.get_exchange(exchange_id)
        if not exchange:
            return {}

        try:
            self.exchange_status[exchange_id or self.primary_exchange].total_calls += 1
            return await asyncio.wait_for(
                asyncio.to_thread(exchange.fetch_tickers, symbols),
                timeout=30.0
            )
        except Exception as exc:
            self.logger.error(f"Failed to fetch tickers: {exc}")
            self.exchange_status[exchange_id or self.primary_exchange].error_count += 1
            return {}

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "5m",
        limit: int = 100,
        exchange_id: str | None = None,
    ) -> list[list[float]] | None:
        exchange = self.get_exchange(exchange_id)
        if not exchange:
            return None

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(exchange.fetch_ohlcv, symbol, timeframe, None, limit),
                timeout=30.0
            )
        except Exception as exc:
            self.logger.error(f"Failed to fetch OHLCV for {symbol}: {exc}")
            return None

    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        exchange_id: str | None = None,
        params: dict | None = None,
    ) -> dict[str, Any] | None:
        exchange = self.get_exchange(exchange_id)
        if not exchange:
            return None

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    exchange.create_order,
                    symbol,
                    order_type,
                    side,
                    amount,
                    price,
                    params or {},
                ),
                timeout=30.0
            )
        except Exception as exc:
            self.logger.error(f"Failed to create order: {exc}")
            return None

    async def fetch_balance(self, exchange_id: str | None = None) -> dict[str, Any]:
        exchange = self.get_exchange(exchange_id)
        if not exchange:
            return {}

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(exchange.fetch_balance),
                timeout=30.0
            )
        except Exception as exc:
            self.logger.error(f"Failed to fetch balance: {exc}")
            return {}

    async def set_leverage(self, leverage: int, symbol: str | None = None, exchange_id: str | None = None) -> dict | None:
        exchange = self.get_exchange(exchange_id)
        if not exchange:
            return None

        try:
            params = {"leverage": leverage}
            if symbol:
                return await asyncio.wait_for(
                    asyncio.to_thread(exchange.set_leverage, leverage, symbol),
                    timeout=10.0
                )
        except Exception as exc:
            self.logger.error(f"Failed to set leverage: {exc}")
            return None

    def get_available_exchanges(self) -> list[str]:
        return list(self.exchanges.keys())

    def get_exchange_info(self, exchange_id: str) -> ExchangeInfo | None:
        return self.exchange_info.get(exchange_id.lower())

    def get_all_exchange_status(self) -> dict[str, dict]:
        return {
            exchange_id: {
                "connected": status.connected,
                "latency_ms": status.latency_ms,
                "last_sync": status.last_sync.isoformat() if status.last_sync else None,
                "error_count": status.error_count,
                "total_calls": status.total_calls,
            }
            for exchange_id, status in self.exchange_status.items()
        }

    def supports_feature(self, exchange_id: str, feature: str) -> bool:
        features = EXCHANGE_FEATURES.get(exchange_id.lower())
        if not features:
            return False
        return features.get(feature, False)


def create_exchange_manager(
    primary_exchange: str = "binance",
    testnet: bool = True,
) -> MultiExchangeManager:
    """Factory function to create and configure a MultiExchangeManager."""
    manager = MultiExchangeManager()
    manager._default_exchange = primary_exchange
    manager.primary_exchange = primary_exchange
    return manager
