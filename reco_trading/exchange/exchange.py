"""
Exchange wrapper for Reco-Trading.
Provides a unified interface for exchange operations.
"""

import logging
from typing import Any

from reco_trading.exchange.binance_client import BinanceClient
from reco_trading.constants import Config


logger = logging.getLogger(__name__)


class Exchange:
    """
    Exchange wrapper that provides a unified interface for trading operations.
    Supports multiple exchanges through CCXT.
    """
    
    def __init__(self, config: Config) -> None:
        """
        Initialize Exchange with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self._config = config
        self._exchange_client: BinanceClient | None = None
        self._name = config.get("exchange", {}).get("name", "binance")
        self._dry_run = config.get("dry_run", True)
        
    async def initialize(self) -> None:
        """Initialize the exchange client."""
        exchange_config = self._config.get("exchange", {})
        
        if self._name.lower() == "binance":
            api_key = exchange_config.get("key", "")
            api_secret = exchange_config.get("secret", "")
            testnet = self._dry_run
            
            self._exchange_client = BinanceClient(api_key, api_secret, testnet)
            
            await self._exchange_client.load_markets()
            await self._exchange_client.sync_time()
            
            logger.info(f"Exchange '{self._name}' initialized (testnet={testnet})")
        else:
            raise ValueError(f"Unsupported exchange: {self._name}")
    
    @property
    def name(self) -> str:
        """Get exchange name."""
        return self._name
    
    @property
    def client(self) -> BinanceClient | None:
        """Get exchange client."""
        return self._exchange_client
    
    @property
    def dry_run(self) -> bool:
        """Check if running in dry-run mode."""
        return self._dry_run
    
    async def fetch_ohlcv(
        self, 
        symbol: str, 
        timeframe: str = "5m", 
        limit: int = 300
    ) -> list:
        """
        Fetch OHLCV data.
        
        Args:
            symbol: Trading pair
            timeframe: Timeframe
            limit: Number of candles
            
        Returns:
            List of OHLCV data
        """
        if self._exchange_client:
            return await self._exchange_client.fetch_ohlcv(symbol, timeframe, limit)
        return []
    
    async def fetch_ticker(self, symbol: str) -> dict | None:
        """
        Fetch ticker data.
        
        Args:
            symbol: Trading pair
            
        Returns:
            Ticker data dictionary
        """
        if self._exchange_client:
            return await self._exchange_client.fetch_ticker(symbol)
        return None
    
    async def fetch_order_book(self, symbol: str, limit: int = 20) -> dict | None:
        """
        Fetch order book.
        
        Args:
            symbol: Trading pair
            limit: Order book depth
            
        Returns:
            Order book dictionary
        """
        if self._exchange_client:
            return await self._exchange_client.fetch_order_book(symbol, limit)
        return None
    
    async def fetch_balance(self) -> dict | None:
        """
        Fetch account balance.
        
        Returns:
            Balance dictionary
        """
        if self._exchange_client:
            return await self._exchange_client.fetch_balance()
        return None
    
    async def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        client_order_id: str | None = None
    ) -> dict | None:
        """
        Create a market order.
        
        Args:
            symbol: Trading pair
            side: Order side ("buy" or "sell")
            amount: Order amount
            client_order_id: Optional client order ID
            
        Returns:
            Order result dictionary
        """
        if self._dry_run:
            logger.info(f"[DRY RUN] Market order: {side} {amount} {symbol}")
            return {
                "id": f"dry_run_{side}_{symbol}",
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "status": "closed",
                "price": 0.0,
            }
        
        if self._exchange_client:
            return await self._exchange_client.create_market_order(
                symbol, side, amount, client_order_id=client_order_id
            )
        return None
    
    async def create_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        client_order_id: str | None = None
    ) -> dict | None:
        """
        Create a limit order.
        
        Args:
            symbol: Trading pair
            side: Order side ("buy" or "sell")
            amount: Order amount
            price: Limit price
            client_order_id: Optional client order ID
            
        Returns:
            Order result dictionary
        """
        if self._dry_run:
            logger.info(f"[DRY RUN] Limit order: {side} {amount} {symbol} @ {price}")
            return {
                "id": f"dry_run_limit_{side}_{symbol}",
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "price": price,
                "status": "open",
            }
        
        if self._exchange_client:
            return await self._exchange_client.create_limit_order(
                symbol, side, amount, price, client_order_id=client_order_id
            )
        return None
    
    async def cancel_order(self, order_id: str, symbol: str) -> dict | None:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID
            symbol: Trading pair
            
        Returns:
            Cancellation result
        """
        if self._dry_run:
            logger.info(f"[DRY RUN] Cancel order: {order_id}")
            return {"id": order_id, "status": "cancelled"}
        
        if self._exchange_client:
            return await self._exchange_client.cancel_order(order_id, symbol)
        return None
    
    async def fetch_order(self, order_id: str, symbol: str) -> dict | None:
        """
        Fetch order status.
        
        Args:
            order_id: Order ID
            symbol: Trading pair
            
        Returns:
            Order status dictionary
        """
        if self._exchange_client:
            return await self._exchange_client.fetch_order(order_id, symbol)
        return None
    
    async def fetch_open_orders(self, symbol: str | None = None) -> list:
        """
        Fetch open orders.
        
        Args:
            symbol: Optional trading pair filter
            
        Returns:
            List of open orders
        """
        if self._exchange_client:
            return await self._exchange_client.fetch_open_orders(symbol)
        return []
    
    def get_pair_quote_currency(self, pair: str) -> str:
        """
        Get quote currency for a pair.
        
        Args:
            pair: Trading pair (e.g., "BTC/USDT")
            
        Returns:
            Quote currency
        """
        if "/" in pair:
            return pair.split("/")[1]
        return ""
    
    def get_pair_base_currency(self, pair: str) -> str:
        """
        Get base currency for a pair.
        
        Args:
            pair: Trading pair (e.g., "BTC/USDT")
            
        Returns:
            Base currency
        """
        if "/" in pair:
            return pair.split("/")[0]
        return ""
    
    async def close(self) -> None:
        """Close exchange connection."""
        logger.info(f"Closing exchange: {self._name}")


def create_exchange(config: Config) -> Exchange:
    """
    Factory function to create an exchange instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Exchange instance
    """
    return Exchange(config)
