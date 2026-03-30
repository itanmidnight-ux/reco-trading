from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import ccxt

logger = logging.getLogger(__name__)


@dataclass
class ExchangeConfig:
    name: str
    api_key: str
    api_secret: str
    testnet: bool = True
    enabled: bool = True


class MultiExchangeManager:
    """
    Manages multiple exchange connections.
    Supports: Binance, Coinbase, Kraken, KuCoin, Bybit, OKX
    """

    SUPPORTED_EXCHANGES = [
        "binance",
        "coinbase", 
        "kraken",
        "kucoin",
        "bybit",
        "okx"
    ]

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.exchanges: dict[str, Any] = {}
        self.primary_exchange: str = "binance"
        self._exchange_configs: dict[str, ExchangeConfig] = {}

    def add_exchange(
        self,
        name: str,
        api_key: str,
        api_secret: str,
        testnet: bool = True
    ) -> bool:
        if name.lower() not in self.SUPPORTED_EXCHANGES:
            self.logger.error(f"Unsupported exchange: {name}")
            return False
            
        try:
            exchange_class = getattr(ccxt, name.lower())
            
            config = {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"}
            }
            
            if testnet:
                if name.lower() == "binance":
                    config["urls"] = {
                        "api": "https://testnet.binance.vision/api",
                        "testnet": "https://testnet.binance.vision/api"
                    }
                elif name.lower() == "kucoin":
                    config["urls"] = {
                        "api": "https://api-sandbox.kucoin.com"
                    }
            
            exchange = exchange_class(config)
            self.exchanges[name.lower()] = exchange
            self._exchange_configs[name.lower()] = ExchangeConfig(
                name=name.lower(),
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet,
                enabled=True
            )
            
            self.logger.info(f"Added exchange: {name} (testnet: {testnet})")
            return True
            
        except Exception as exc:
            self.logger.error(f"Failed to add exchange {name}: {exc}")
            return False

    def set_primary(self, name: str) -> bool:
        if name.lower() in self.exchanges:
            self.primary_exchange = name.lower()
            self.logger.info(f"Primary exchange set to: {name}")
            return True
        return False

    def get_primary(self) -> Any:
        return self.exchanges.get(self.primary_exchange)

    def get_exchange(self, name: str | None = None) -> Any:
        if name:
            return self.exchanges.get(name.lower())
        return self.get_primary()

    async def fetch_tickers(self, symbols: list[str] | None = None) -> dict[str, Any]:
        exchange = self.get_primary()
        if not exchange:
            return {}
            
        try:
            tickers = await asyncio.wait_for(
                asyncio.to_thread(exchange.fetch_tickers, symbols),
                timeout=30.0
            )
            return tickers
        except Exception as exc:
            self.logger.error(f"Failed to fetch tickers: {exc}")
            return {}

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "5m",
        limit: int = 100,
        exchange_name: str | None = None
    ) -> list[list[float]] | None:
        exchange = self.get_exchange(exchange_name)
        if not exchange:
            return None
            
        try:
            ohlcv = await asyncio.wait_for(
                asyncio.to_thread(
                    exchange.fetch_ohlcv,
                    symbol,
                    timeframe,
                    None,
                    limit
                ),
                timeout=30.0
            )
            return ohlcv
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
        exchange_name: str | None = None,
        params: dict | None = None
    ) -> dict[str, Any] | None:
        exchange = self.get_exchange(exchange_name)
        if not exchange:
            return None
            
        try:
            order = await asyncio.wait_for(
                asyncio.to_thread(
                    exchange.create_order,
                    symbol,
                    order_type,
                    side,
                    amount,
                    price,
                    params or {}
                ),
                timeout=30.0
            )
            return order
        except Exception as exc:
            self.logger.error(f"Failed to create order: {exc}")
            return None

    def get_enabled_exchanges(self) -> list[str]:
        return [name for name, config in self._exchange_configs.items() if config.enabled]

    def get_exchange_status(self) -> dict[str, dict[str, Any]]:
        status = {}
        for name, config in self._exchange_configs.items():
            status[name] = {
                "enabled": config.enabled,
                "testnet": config.testnet,
                "connected": name in self.exchanges,
            }
        return status
