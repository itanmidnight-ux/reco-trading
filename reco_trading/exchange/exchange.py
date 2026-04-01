"""
Exchange wrapper for Reco-Trading.
Provides a unified interface for exchange operations.
"""

import logging
from typing import Any

from reco_trading.exchange.binance_client import BinanceClient as RobustBinanceClient


logger = logging.getLogger(__name__)


class Exchange:
    """
    Exchange wrapper that provides a unified interface for trading operations.
    Supports multiple exchanges through CCXT.
    """
    
    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        exchange_name: str = "binance",
        testnet: bool = True,
        trading_mode: str = "spot",
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.api_secret = api_secret
        self.exchange_name = exchange_name.lower()
        self.testnet = testnet
        self.trading_mode = trading_mode
        self._exchange_client: RobustBinanceClient | None = None
        self._connected = False
        
    async def initialize(self) -> bool:
        """Initialize the exchange client."""
        try:
            if self.exchange_name == "binance":
                self._exchange_client = RobustBinanceClient(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    testnet=self.testnet,
                    trading_mode=self.trading_mode,
                )
                
                await self._exchange_client.connect()
                self._connected = True
                
                self.logger.info(
                    f"Exchange '{self.exchange_name}' initialized "
                    f"(testnet={self.testnet}, mode={self.trading_mode})"
                )
                return True
            else:
                raise ValueError(f"Unsupported exchange: {self.exchange_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize exchange: {e}")
            self._connected = False
            raise
    
    @property
    def name(self) -> str:
        """Get exchange name."""
        return self.exchange_name
    
    @property
    def client(self) -> RobustBinanceClient | None:
        """Get exchange client."""
        return self._exchange_client
    
    @property
    def connected(self) -> bool:
        """Check if connected."""
        return self._connected
    
    @property
    def dry_run(self) -> bool:
        """Check if running in testnet mode."""
        return self.testnet
    
    async def fetch_ohlcv(
        self, 
        symbol: str, 
        timeframe: str = "5m", 
        limit: int = 300
    ) -> list[list[float]]:
        """Fetch OHLCV data."""
        if self._exchange_client:
            return await self._exchange_client.fetch_ohlcv(symbol, timeframe, limit)
        return []
    
    async def fetch_ticker(self, symbol: str) -> dict[str, Any] | None:
        """Fetch ticker data."""
        if self._exchange_client:
            return await self._exchange_client.fetch_ticker(symbol)
        return None
    
    async def fetch_order_book(self, symbol: str, limit: int = 20) -> dict[str, Any] | None:
        """Fetch order book."""
        if self._exchange_client:
            return await self._exchange_client.fetch_order_book(symbol, limit)
        return None
    
    async def fetch_balance(self) -> dict[str, Any] | None:
        """Fetch account balance."""
        if self._exchange_client:
            return await self._exchange_client.fetch_balance()
        return None
    
    async def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        client_order_id: str | None = None
    ) -> dict[str, Any] | None:
        """Create a market order."""
        if self.testnet:
            self.logger.info(f"[DRY RUN] Market order: {side} {amount} {symbol}")
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
    ) -> dict[str, Any] | None:
        """Create a limit order."""
        if self.testnet:
            self.logger.info(f"[DRY RUN] Limit order: {side} {amount} {symbol} @ {price}")
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
    
    async def cancel_order(self, order_id: str, symbol: str) -> dict[str, Any] | None:
        """Cancel an order."""
        if self.testnet:
            self.logger.info(f"[DRY RUN] Cancel order: {order_id}")
            return {"id": order_id, "status": "cancelled"}
        
        if self._exchange_client:
            return await self._exchange_client.cancel_order(order_id, symbol)
        return None
    
    async def fetch_order(self, order_id: str, symbol: str) -> dict[str, Any] | None:
        """Fetch order status."""
        if self._exchange_client:
            return await self._exchange_client.fetch_order(order_id, symbol)
        return None
    
    async def fetch_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Fetch open orders."""
        if self._exchange_client:
            return await self._exchange_client.fetch_open_orders(symbol)
        return []
    
    async def fetch_my_trades(self, symbol: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Fetch trade history."""
        if self._exchange_client:
            return await self._exchange_client.fetch_my_trades(symbol, limit)
        return []
    
    def get_pair_quote_currency(self, pair: str) -> str:
        """Get quote currency for a pair."""
        if "/" in pair:
            return pair.split("/")[1]
        return ""
    
    def get_pair_base_currency(self, pair: str) -> str:
        """Get base currency for a pair."""
        if "/" in pair:
            return pair.split("/")[0]
        return ""
    
    def get_status(self) -> dict[str, Any]:
        """Get exchange status."""
        if self._exchange_client:
            return self._exchange_client.get_status()
        return {
            "exchange": self.exchange_name,
            "connected": self._connected,
            "testnet": self.testnet,
        }
    
    async def close(self) -> None:
        """Close exchange connection."""
        if self._exchange_client:
            await self._exchange_client.close()
            self._connected = False
            self.logger.info(f"Exchange '{self.exchange_name}' closed")


def create_exchange(
    api_key: str = "",
    api_secret: str = "",
    exchange_name: str = "binance",
    testnet: bool = True,
    trading_mode: str = "spot",
) -> Exchange:
    """Factory function to create an exchange instance."""
    return Exchange(
        api_key=api_key,
        api_secret=api_secret,
        exchange_name=exchange_name,
        testnet=testnet,
        trading_mode=trading_mode,
    )
