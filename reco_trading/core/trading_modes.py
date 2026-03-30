from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TradingMode:
    """Trading mode configuration."""
    mode: str = "spot"
    leverage: int = 1
    side: str = "long"
    hedge_mode: bool = False


@dataclass 
class FuturesConfig:
    """Configuration for futures trading."""
    enabled: bool = False
    trading_mode: str = "spot"
    leverage: int = 1
    default_side: str = "long"
    hedge_mode: bool = False
    max_leverage: int = 10
    liquidation_buffer: float = 0.05
    funding_fee_management: bool = True


class TradingModeManager:
    """
    Manages trading modes: Spot, Margin, Futures with Long/Short support.
    """

    def __init__(self, exchange_client: Any, config: FuturesConfig | None = None):
        self.logger = logging.getLogger(__name__)
        self.exchange = exchange_client
        self.config = config or FuturesConfig()
        self.current_mode = TradingMode(
            mode=self.config.trading_mode,
            leverage=self.config.leverage,
            side=self.config.default_side,
            hedge_mode=self.config.hedge_mode,
        )
        self._position_info: dict | None = None

    async def set_mode(self, mode: str, leverage: int = 1) -> bool:
        """Set trading mode (spot, margin, futures)."""
        
        valid_modes = ["spot", "margin", "futures"]
        if mode not in valid_modes:
            self.logger.error(f"Invalid trading mode: {mode}")
            return False
        
        if mode == "futures" and leverage > self.config.max_leverage:
            self.logger.warning(f"Leverage {leverage} exceeds max {self.config.max_leverage}, using max")
            leverage = self.config.max_leverage
        
        try:
            if mode == "futures" and hasattr(self.exchange, "exchange"):
                ccxt_exchange = self.exchange.exchange
                if hasattr(ccxt_exchange, "set_leverage"):
                    await asyncio.to_thread(ccxt_exchange.set_leverage, leverage)
            
            self.current_mode.mode = mode
            self.current_mode.leverage = leverage
            
            self.logger.info(f"Trading mode set to: {mode} (leverage: {leverage}x)")
            return True
            
        except Exception as exc:
            self.logger.error(f"Failed to set trading mode: {exc}")
            return False

    async def set_side(self, side: str) -> bool:
        """Set trading side (long or short)."""
        
        if self.current_mode.mode == "spot":
            self.logger.info("Spot mode: side is always LONG")
            side = "long"
        
        valid_sides = ["long", "short"]
        if side not in valid_sides:
            self.logger.error(f"Invalid side: {side}")
            return False
        
        self.current_mode.side = side
        self.logger.info(f"Trading side set to: {side}")
        return True

    def get_entry_side(self) -> str:
        """Get entry side based on current mode."""
        
        if self.current_mode.mode == "spot":
            return "BUY"
        
        if self.current_mode.side == "long":
            return "BUY"
        else:
            return "SELL"

    def get_exit_side(self) -> str:
        """Get exit side based on current mode."""
        
        if self.current_mode.mode == "spot":
            return "SELL"
        
        if self.current_mode.side == "long":
            return "SELL"
        else:
            return "BUY"

    async def get_position(self, symbol: str) -> dict | None:
        """Get current position info."""
        
        if self.current_mode.mode != "futures":
            return None
        
        try:
            ccxt_exchange = getattr(self.exchange, "exchange", self.exchange)
            if hasattr(ccxt_exchange, "fetch_positions"):
                positions = await asyncio.to_thread(ccxt_exchange.fetch_positions, [symbol])
                for pos in positions:
                    if pos.get("symbol") == symbol:
                        self._position_info = pos
                        return pos
        except Exception as exc:
            self.logger.warning(f"Failed to get position: {exc}")
        
        return None

    def calculate_liquidation_price(
        self,
        entry_price: float,
        side: str,
        leverage: int,
        balance: float,
        position_size: float,
    ) -> float | None:
        """Calculate liquidation price for futures position."""
        
        if self.current_mode.mode != "futures" or leverage == 1:
            return None
        
        try:
            maintenance_margin_rate = 0.005
            buffer = self.config.liquidation_buffer
            
            if side.lower() == "long":
                liquidation_price = entry_price * (1 - (1 / leverage) + maintenance_margin_rate + buffer)
            else:
                liquidation_price = entry_price * (1 + (1 / leverage) - maintenance_margin_rate - buffer)
            
            return liquidation_price
            
        except Exception as exc:
            self.logger.error(f"Failed to calculate liquidation price: {exc}")
            return None

    def get_status(self) -> dict:
        """Get current trading mode status."""
        
        return {
            "mode": self.current_mode.mode,
            "leverage": self.current_mode.leverage,
            "side": self.current_mode.side,
            "hedge_mode": self.current_mode.hedge_mode,
            "is_futures": self.current_mode.mode == "futures",
            "is_margin": self.current_mode.mode == "margin",
        }


class WebSocketManager:
    """
    WebSocket manager for real-time data streaming.
    """

    def __init__(self, exchange_client: Any):
        self.logger = logging.getLogger(__name__)
        self.exchange = exchange_client
        self._ws_connection: Any = None
        self._streams: dict[str, asyncio.Queue] = {}
        self._is_connected: bool = False

    async def connect(self, streams: list[str]) -> bool:
        """Connect to WebSocket streams."""
        
        try:
            ccxt_exchange = getattr(self.exchange, "exchange", self.exchange)
            
            if hasattr(ccxt_exchange, "watch_ticker"):
                for symbol in streams:
                    self._streams[symbol] = asyncio.Queue()
                
                self._is_connected = True
                self.logger.info(f"WebSocket connected for {len(streams)} streams")
                return True
            else:
                self.logger.warning("Exchange does not support WebSocket")
                return False
                
        except Exception as exc:
            self.logger.error(f"WebSocket connection failed: {exc}")
            return False

    async def subscribe(self, symbol: str, callback: callable) -> bool:
        """Subscribe to a symbol's stream."""
        
        if symbol not in self._streams:
            self.logger.warning(f"Stream not found for {symbol}")
            return False
        
        asyncio.create_task(self._listen_stream(symbol, callback))
        return True

    async def _listen_stream(self, symbol: str, callback: callable) -> None:
        """Listen to a stream and process messages."""
        
        while self._is_connected:
            try:
                ccxt_exchange = getattr(self.exchange, "exchange", self.exchange)
                
                if hasattr(ccxt_exchange, "watch_ticker"):
                    ticker = await asyncio.to_thread(ccxt_exchange.watch_ticker, symbol)
                    await callback(ticker)
                    
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.logger.error(f"Stream error for {symbol}: {exc}")
                await asyncio.sleep(1)

    async def disconnect(self) -> None:
        """Disconnect WebSocket."""
        
        self._is_connected = False
        
        for stream in self._streams.values():
            stream.put_nowait(None)
        
        self._streams.clear()
        self.logger.info("WebSocket disconnected")

    def is_connected(self) -> bool:
        return self._is_connected


def create_hyperopt_config(
    epochs: int = 100,
    max_trades: int = 3,
    spaces: list[str] | None = None,
    loss_function: str = "sharpe",
) -> dict:
    """Create hyperopt configuration."""
    
    default_spaces = ["buy", "sell", "roi", "stoploss", "trailing"]
    
    return {
        "epochs": epochs,
        "max_trades": max_trades,
        "spaces": spaces or default_spaces,
        "loss_function": loss_function,
        "early_stop": True,
        "early_stop_minutes": 60,
        "parallel": True,
        "n_jobs": -1,
    }
