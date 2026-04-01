#!/usr/bin/env python3
"""
Enhanced Multi-Exchange Support
Adds support for 20+ exchanges with auto-detection and intelligent routing
"""
from __future__ import annotations

import asyncio
import ccxt
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

from .exchange import Exchange
from ..config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass 
class ExchangeConfig:
    """Exchange configuration with intelligent defaults"""
    exchange_id: str
    api_key: str = ""
    api_secret: str = ""
    passphrase: Optional[str] = None
    testnet: bool = True
    trading_mode: str = "spot"  # spot, margin, futures
    leverage: int = 1
    enabled: bool = True
    
    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_secret)


class ExchangeRouter:
    """
    Intelligent exchange routing system for optimal execution
    """
    
    def __init__(self, exchange_manager: 'EnhancedMultiExchange'):
        self.manager = exchange_manager
        self.routing_table: Dict[str, str] = {}
        
    async def get_best_exchange_for_symbol(self, symbol: str) -> str:
        """Get the best exchange for a specific symbol based on liquidity and fees"""
        best_exchange = None
        best_score = -1
        
        for exchange_id in self.manager.get_available_exchanges():
            try:
                tickers = await self.manager.fetch_tickers([symbol], exchange_id)
                if symbol in tickers:
                    # Score based on 24h volume and spread
                    ticker = tickers[symbol]
                    volume = ticker.get('quoteVolume', 0)
                    spread = ticker.get('percentage', 0)
                    
                    # Normalize score (higher is better)
                    score = volume * (1 - abs(spread))
                    if score > best_score:
                        best_score = score
                        best_exchange = exchange_id
                        
            except Exception as e:
                logger.debug(f"Failed to fetch ticker for {symbol} on {exchange_id}: {e}")
                
        return best_exchange or self.manager.primary_exchange


class EnhancedMultiExchange:
    """
    Enhanced Multi-Exchange Manager with 20+ supported exchanges
    Advanced features: intelligent routing, auto-failover, fee optimization
    """
    
    # Advanced exchange configurations with detailed features
    EXCHANGE_CONFIGS = {
        "binance": {
            "name": "Binance",
            "countries": ["US", "Global"], 
            "fees": {"maker": 0.001, "taker": 0.001},
            "features": ["spot", "margin", "futures", "options", "leverage"],
            "testnet": True,
            "api": ccxt.binance,
        },
        "coinbase": {
            "name": "Coinbase",
            "countries": ["US"],
            "fees": {"maker": 0.005, "taker": 0.005},
            "features": ["spot", "futures"],
            "testnet": False,
            "api": ccxt.coinbase,
        },
        "kraken": {
            "name": "Kraken", 
            "countries": ["US", "Global"],
            "fees": {"maker": 0.0026, "taker": 0.0026},
            "features": ["spot", "margin", "futures"],
            "testnet": False,
            "api": ccxt.kraken,
        },
        "kucoin": {
            "name": "KuCoin",
            "countries": ["Global"],
            "fees": {"maker": 0.001, "taker": 0.001},
            "features": ["spot", "margin", "futures"],
            "testnet": True,
            "api": ccxt.kucoin,
        },
        "bybit": {
            "name": "Bybit",
            "countries": ["Global"],
            "fees": {"maker": 0.001, "taker": 0.001},
            "features": ["spot", "margin", "futures"],
            "testnet": True,
            "api": ccxt.bybit,
        },
        "okx": {
            "name": "OKX",
            "countries": ["Global", "US"],
            "fees": {"maker": 0.0008, "taker": 0.001},
            "features": ["spot", "margin", "futures"],
            "testnet": False,
            "api": ccxt.okx,
        },
        "gate": {
            "name": "Gate.io",
            "countries": ["Global"],
            "fees": {"maker": 0.002, "taker": 0.002},
            "features": ["spot", "margin", "futures"],
            "testnet": True,
            "api": ccxt.gateio,
        },
        "bitget": {
            "name": "Bitget",
            "countries": ["Global"],
            "fees": {"maker": 0.001, "taker": 0.001},
            "features": ["spot", "margin", "futures"],
            "testnet": True,
            "api": ccxt.bitget,
        },
        "huobi": {
            "name": "Huobi",
            "countries": ["Global"],
            "fees": {"maker": 0.002, "taker": 0.002},
            "features": ["spot", "margin", "futures"],
            "testnet": True,
            "api": ccxt.huobi,
        },
        "mexc": {
            "name": "MEXC",
            "countries": ["Global"],
            "fees": {"maker": 0.001, "taker": 0.001},
            "features": ["spot", "margin", "futures"],
            "testnet": False,
            "api": ccxt.mexc,
        },
        "kucoinfutures": {
            "name": "KuCoin Futures",
            "countries": ["Global"],
            "fees": {"maker": 0.0002, "taker": 0.0004},
            "features": ["futures"],
            "testnet": True,
            "api": ccxt.kucoinfutures,
        },
        "bitstamp": {
            "name": "Bitstamp",
            "countries": ["US", "Global"],
            "fees": {"maker": 0.003, "taker": 0.005},
            "features": ["spot"],
            "testnet": False,
            "api": ccxt.bitstamp,
        },
        "cryptocom": {
            "name": "Crypto.com",
            "countries": ["Global"],
            "fees": {"maker": 0.0004, "taker": 0.001},
            "features": ["spot", "margin", "futures"],
            "testnet": False,
            "api": ccxt.cryptocom,
        },
        "phemex": {
            "name": "Phemex",
            "countries": ["Global"],
            "fees": {"maker": 0.0001, "taker": 0.0004},
            "features": ["spot", "futures"],
            "testnet": True,
            "api": ccxt.phemex,
        },
        "bingx": {
            "name": "BingX",
            "countries": ["Global"],
            "fees": {"maker": 0.001, "taker": 0.001},
            "features": ["spot", "margin", "futures"],
            "testnet": True, 
            "api": ccxt.bingx,
        },
        "bitfinex": {
            "name": "Bitfinex",
            "countries": ["Global"],
            "fees": {"maker": 0.001, "taker": 0.001},
            "features": ["spot", "margin", "futures"],
            "testnet": False,
            "api": ccxt.bitfinex,
        },
        "htx": {
            "name": "HTX (Huobi)",
            "countries": ["Global"],
            "fees": {"maker": 0.002, "taker": 0.002},
            "features": ["spot", "margin", "futures"],
            "testnet": True,
            "api": ccxt.htx,
        },
        "woo": {
            "name": "WOO Network",
            "countries": ["Global"],
            "fees": {"maker": 0.0, "taker": 0.001},
            "features": ["spot"],
            "testnet": True,
            "api": ccxt.woo,
        },
        "deribit": {
            "name": "Deribit",
            "countries": ["Global"],
            "fees": {"maker": 0.0003, "taker": 0.0003},
            "features": ["spot", "futures", "options"],
            "testnet": True,
            "api": ccxt.deribit,
        },
        "dydx": {
            "name": "dYdX",
            "countries": ["Global"],
            "fees": {"maker": 0.0005, "taker": 0.0005},
            "features": ["futures"],
            "testnet": True,
            "api": ccxt.dydx,
        },
        "hyperliquid": {
            "name": "Hyperliquid",
            "countries": ["Global"],
            "fees": {"maker": 0.0002, "taker": 0.0005},
            "features": ["futures"],
            "testnet": True,
            "api": ccxt.hyperliquid,
        }
    }
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.exchange_configs: Dict[str, ExchangeConfig] = {}
        self.primary_exchange: str = "binance"
        self.router = ExchangeRouter(self)
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self) -> None:
        """Initialize all configured exchanges"""
        await self.auto_detect_exchanges()
        await self.connect_all()
        
    async def auto_detect_exchanges(self) -> None:
        """Auto-detect exchanges from environment variables"""
        detected_exchanges = []
        
        for exchange_id, config in self.EXCHANGE_CONFIGS.items():
            # Check environment variables for API keys
            api_key = getattr(self.settings, f"{exchange_id}_api_key", None)
            api_secret = getattr(self.settings, f"{exchange_id}_api_secret", None)
            passphrase = getattr(self.settings, f"{exchange_id}_passphrase", None)
            
            if api_key and api_secret:
                exchange_config = ExchangeConfig(
                    exchange_id=exchange_id,
                    api_key=api_key,
                    api_secret=api_secret,
                    passphrase=passphrase,
                    testnet=getattr(self.settings, f"{exchange_id}_testnet", True),
                    trading_mode=getattr(self.settings, f"{exchange_id}_mode", "spot"),
                    leverage=getattr(self.settings, f"{exchange_id}_leverage", 1),
                )
                
                self.exchange_configs[exchange_id] = exchange_config
                detected_exchanges.append(exchange_id)
                self.logger.info(f"Auto-detected exchange: {exchange_id}")
        
        # Set primary exchange (Binance by default, or first detected)
        if "binance" in self.exchange_configs:
            self.primary_exchange = "binance"
        elif detected_exchanges:
            self.primary_exchange = detected_exchanges[0]
            
        self.logger.info(f"Primary exchange set to: {self.primary_exchange}")
        
    async def connect_all(self) -> None:
        """Connect all configured exchanges"""
        connection_tasks = []
        
        for exchange_id, config in self.exchange_configs.items():
            if config.enabled and config.is_configured:
                connection_tasks.append(self._connect_exchange(exchange_id, config))
                
        results = await asyncio.gather(*connection_tasks, return_exceptions=True)
        
        for exchange_id, result in zip(self.exchange_configs.keys(), results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to connect {exchange_id}: {result}")
                
    async def _connect_exchange(self, exchange_id: str, config: ExchangeConfig) -> bool:
        """Connect to a specific exchange"""
        try:
            exchange_config = self.EXCHANGE_CONFIGS[exchange_id]
            exchange_class = exchange_config["api"]
            
            # Build exchange configuration
            exchange_params = {
                "apiKey": config.api_key,
                "secret": config.api_secret,
                "enableRateLimit": True,
                "timeout": 30000,
            }
            
            # Add passphrase for exchanges that need it (like KuCoin)
            if config.passphrase:
                exchange_params["password"] = config.passphrase
                
            # Configure testnet URLs
            if config.testnet:
                testnet_urls = {
                    "binance": "https://testnet.binance.vision/api",
                    "kucoin": "https://api-sandbox.kucoin.com", 
                    "bybit": "https://api-testnet.bybit.com",
                    "gate": "https://api-testnet.gateio.ws",
                    "bitget": "https://api-testnet.bitget.com",
                    "huobi": "https://api.huobi.pro",
                    "bingx": "https://open-api.bingx.pro",
                    "phemex": "https://testnet-api.phemex.com",
                    "woo": "https://testnet.woox.io",
                    "deribit": "https://testnet.deribit.com",
                    "dydx": "https://api.dydx.trade",
                    "hyperliquid": "https://api.hyperliquid-testnet.xyz",
                }
                
                if exchange_id in testnet_urls:
                    exchange_params["urls"] = {"api": testnet_urls[exchange_id]}
                    
            # Configure trading mode
            if config.trading_mode == "futures":
                exchange_params["options"] = {"defaultType": "future"}
            elif config.trading_mode == "margin":
                exchange_params["options"] = {"defaultType": "margin"}
                
            # Create exchange instance
            exchange = exchange_class(exchange_params)
            
            # Test connection
            await asyncio.wait_for(asyncio.to_thread(exchange.fetch_time), timeout=10.0)
            
            self.exchanges[exchange_id] = exchange
            self.logger.info(f"Connected to {exchange_id} (mode: {config.trading_mode})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect {exchange_id}: {e}")
            return False
            
    async def smart_order_routing(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Smart order routing to get best execution"""
        
        # Get best exchange for this symbol
        best_exchange = await self.router.get_best_exchange_for_symbol(symbol)
        
        # Check if primary exchange has better fees
        primary_fees = self.EXCHANGE_CONFIGS[self.primary_exchange]["fees"]
        best_fees = self.EXCHANGE_CONFIGS[best_exchange]["fees"]
        
        # Use primary if fees are significantly better
        fee_diff = abs(primary_fees["taker"] - best_fees["taker"])
        if fee_diff < 0.0001:  # Difference less than 0.01%
            best_exchange = self.primary_exchange
            
        # Execute order on best exchange
        order = await self.create_order(
            symbol, side, order_type, amount, price, best_exchange
        )
        
        return best_exchange, order
        
    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
        exchange_id: Optional[str] = None,
        params: Optional[Dict] = None,
    ) -> Optional[Dict[str, Any]]:
        """Create order with automatic exchange selection"""
        if exchange_id and exchange_id in self.exchanges:
            exchange = self.exchanges[exchange_id]
        else:
            exchange = self.exchanges.get(self.primary_exchange)
            
        if not exchange:
            raise ValueError(f"No exchange available for symbol {symbol}")
            
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
        except Exception as e:
            self.logger.error(f"Order failed on {exchange.id}: {e}")
            # Try failover to another exchange
            for fallback_id, fallback_exchange in self.exchanges.items():
                if fallback_id != exchange.id:
                    try:
                        self.logger.info(f"Failing over to {fallback_id}")
                        return await asyncio.wait_for(
                            asyncio.to_thread(
                                fallback_exchange.create_order,
                                symbol,
                                order_type,
                                side,
                                amount,
                                price,
                                params or {},
                            ),
                            timeout=30.0
                        )
                    except Exception:
                        continue
            raise
            
    async def fetch_tickers(
        self,
        symbols: Optional[List[str]] = None,
        exchange_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch tickers from all or specific exchange"""
        if exchange_id and exchange_id in self.exchanges:
            exchange = self.exchanges[exchange_id]
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(exchange.fetch_tickers, symbols),
                    timeout=30.0
                )
            except Exception as e:
                self.logger.error(f"Failed to fetch tickers from {exchange_id}: {e}")
                return {}
                
        # Aggregate tickers from all exchanges
        aggregated_tickers = {}
        tasks = []
        
        for ex_id, exchange in self.exchanges.items():
            tasks.append(asyncio.create_task(
                self._fetch_exchange_tickers(ex_id, exchange, symbols)
            ))
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, dict):
                aggregated_tickers.update(result)
                
        return aggregated_tickers
        
    async def _fetch_exchange_tickers(
        self,
        exchange_id: str,
        exchange: ccxt.Exchange,
        symbols: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Helper to fetch tickers from specific exchange"""
        try:
            tickers = await asyncio.wait_for(
                asyncio.to_thread(exchange.fetch_tickers, symbols),
                timeout=30.0
            )
            # Prefix with exchange ID to avoid conflicts
            return {f"{exchange_id}:{symbol}": ticker for symbol, ticker in tickers.items()}
        except Exception as e:
            self.logger.debug(f"Failed to fetch tickers from {exchange_id}: {e}")
            return {}
            
    def get_available_exchanges(self) -> List[str]:
        """Get list of connected exchanges"""
        return list(self.exchanges.keys())
        
    def get_exchange(self, exchange_id: Optional[str] = None) -> Optional[ccxt.Exchange]:
        """Get specific exchange or primary"""
        if exchange_id:
            return self.exchanges.get(exchange_id)
        return self.exchanges.get(self.primary_exchange)
        
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get aggregated portfolio across all exchanges"""
        total_balance = {}
        exchange_balances = {}
        
        for exchange_id, exchange in self.exchanges.items():
            try:
                balance = await asyncio.wait_for(
                    asyncio.to_thread(exchange.fetch_balance),
                    timeout=30.0
                )
                exchange_balances[exchange_id] = balance
                
                # aggregate free balance
                for currency, amounts in balance.get('free', {}).items():
                    if amounts > 0:
                        total_balance[currency] = total_balance.get(currency, 0) + amounts
                        
            except Exception as e:
                self.logger.error(f"Failed to fetch balance from {exchange_id}: {e}")
                
        return {
            "total_balance": total_balance,
            "exchange_balances": exchange_balances,
            "connected_exchanges": len(self.exchanges),
            "primary_exchange": self.primary_exchange,
        }