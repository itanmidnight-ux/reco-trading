#!/usr/bin/env python3
"""
Enhanced Grid Trading System
Professional grid trading with adaptive grid spacing, dynamic rebalancing, and intelligent risk management
"""
from __future__ import annotations

import asyncio
import logging
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import numpy as np
import pandas as pd

from ..config.settings import Settings

logger = logging.getLogger(__name__)


class GridDirection(Enum):
    """Grid trading direction"""
    BOTH = "both"      # Buy low, sell high
    LONG = "long"      # Buy dips only
    SHORT = "short"    # Sell rallies only


class GridType(Enum):
    """Grid calculation method"""
    FIXED = "fixed"                # Fixed grid spacing
    PERCENTAGE = "percentage"      # Percentage based
    VOLATILITY = "volatility"      # ATR-based dynamic
    FIBONACCI = "fibonacci"        # Fibonacci retracement
    SUPPORT_RESISTANCE = "sr"      # Support/resistance levels


@dataclass
class GridLevel:
    """Single grid level configuration"""
    price: float
    order_type: str  # "buy" or "sell"
    amount: float
    placed: bool = False
    order_id: Optional[str] = None
    filled: bool = False
    fill_time: Optional[datetime] = None
    
    @property
    def is_active(self) -> bool:
        return self.placed and not self.filled


@dataclass
class GridConfig:
    """Grid trading configuration"""
    symbol: str
    direction: GridDirection = GridDirection.BOTH
    grid_type: GridType = GridType.PERCENTAGE
    center_price: Optional[float] = None
    price_range: float = 0.05  # 5% total range
    num_grids: int = 10
    amount_per_grid: float = 10.0
    leverage: int = 1
    
    # Dynamic parameters
    enable_dynamic_spacing: bool = True
    enable_rebalancing: bool = True
    rebalance_threshold: float = 0.02  # 2% price move triggers rebalance
    
    # Risk management
    max_total_investment: float = 1000.0
    stop_loss_percentage: float = 0.10  # 10%
    take_profit_grid_count: int = 3  # Take profit after N profitable grids


class AdvancedGridTrader:
    """
    Advanced grid trading system with dynamic adaptation
    Features: ATR-based spacing, market condition detection, smart positioning
    """
    
    def __init__(self, exchange, settings: Settings):
        self.exchange = exchange
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Grid tracking
        self.active_grids: Dict[str, List[GridLevel]] = {}
        self.grid_configs: Dict[str, GridConfig] = {}
        
        # Market analysis
        self.volatility_history: Dict[str, pd.Series] = {}
        self.support_resistance_levels: Dict[str, Tuple[float, float]] = {}
        
    async def initialize_grid(self, config: GridConfig) -> bool:
        """Initialize a new grid trading setup"""
        try:
            # Get current市场价格
            ticker = await asyncio.wait_for(
                asyncio.to_thread(self.exchange.fetch_ticker, config.symbol),
                timeout=10.0
            )
            current_price = ticker['last']
            
            # Set center price if not provided
            if not config.center_price:
                config.center_price = current_price
                
            # Generate grid levels
            grid_levels = await self._generate_grid_levels(config, current_price)
            
            # Store grid
            self.grid_configs[config.symbol] = config
            self.active_grids[config.symbol] = grid_levels
            
            # Place initial grid orders
            await self._place_grid_orders(config.symbol)
            
            self.logger.info(f"Grid initialized for {config.symbol} with {len(grid_levels)} levels")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize grid: {e}")
            return False
            
    async def _generate_grid_levels(self, config: GridConfig, current_price: float) -> List[GridLevel]:
        """Generate grid levels based on configuration"""
        
        if config.grid_type == GridType.FIXED:
            return self._generate_fixed_grid(config, current_price)
        elif config.grid_type == GridType.PERCENTAGE:
            return self._generate_percentage_grid(config, current_price)
        elif config.grid_type == GridType.VOLATILITY:
            return self._generate_volatility_grid(config, current_price)
        elif config.grid_type == GridType.FIBONACCI:
            return self._generate_fibonacci_grid(config, current_price)
        elif config.grid_type == GridType.SUPPORT_RESISTANCE:
            return self._generate_support_resistance_grid(config, current_price)
        else:
            return self._generate_percentage_grid(config, current_price)
            
    def _generate_fixed_grid(self, config: GridConfig, current_price: float) -> List[GridLevel]:
        """Generate fixed spacing grid"""
        grid_spacing = current_price * config.price_range / (config.num_grids - 1) / 2
        levels = []
        
        # Generate buy levels (below center)
        for i in range(config.num_grids // 2):
            price = config.center_price - (i + 1) * grid_spacing
            if config.direction in [GridDirection.BOTH, GridDirection.LONG]:
                levels.append(GridLevel(
                    price=price,
                    order_type="buy",
                    amount=config.amount_per_grid
                ))
                
        # Generate sell levels (above center)
        for i in range(config.num_grids // 2):
            price = config.center_price + (i + 1) * grid_spacing
            if config.direction in [GridDirection.BOTH, GridDirection.SHORT]:
                levels.append(GridLevel(
                    price=price,
                    order_type="sell",
                    amount=config.amount_per_grid
                ))
                
        return levels
        
    def _generate_percentage_grid(self, config: GridConfig, current_price: float) -> List[GridLevel]:
        """Generate percentage-based grid"""
        levels = []
        half_range = config.price_range / 2
        
        # Buy levels (below center)
        for i in range(1, config.num_grids // 2 + 1):
            percentage = (i / (config.num_grids // 2)) * half_range
            price = config.center_price * (1 - percentage)
            
            if config.direction in [GridDirection.BOTH, GridDirection.LONG]:
                levels.append(GridLevel(
                    price=price,
                    order_type="buy",
                    amount=config.amount_per_grid
                ))
                
        # Sell levels (above center)
        for i in range(1, config.num_grids // 2 + 1):
            percentage = (i / (config.num_grids // 2)) * half_range
            price = config.center_price * (1 + percentage)
            
            if config.direction in [GridDirection.BOTH, GridDirection.SHORT]:
                levels.append(GridLevel(
                    price=price,
                    order_type="sell",
                    amount=config.amount_per_grid
                ))
                
        return levels
        
    async def _generate_volatility_grid(self, config: GridConfig, current_price: float) -> List[GridLevel]:
        """Generate dynamic grid based on ATR (volatility)"""
        # Fetch recent OHLCV data for volatility calculation
        ohlcv = await asyncio.wait_for(
            asyncio.to_thread(
                self.exchange.fetch_ohlcv, config.symbol, "1h", None, 100
            ),
            timeout=30.0
        )
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Calculate ATR (Average True Range)
        import talib
        atr = talib.ATR(df['high'].values, df['low'].values, df['close'].values, 14)[-1]
        atr_percentage = atr / current_price
        
        # Use ATR for dynamic grid spacing
        levels = []
        grid_multiplier = 0.5  # Use ATR as base spacing multiplier
        
        # Generate levels using ATR-based spacing
        for i in range(1, config.num_grids // 2 + 1):
            spacing = atr * grid_multiplier * i
            
            # Buy level
            buy_price = config.center_price - spacing
            if config.direction in [GridDirection.BOTH, GridDirection.LONG]:
                levels.append(GridLevel(
                    price=buy_price,
                    order_type="buy",
                    amount=config.amount_per_grid
                ))
                
            # Sell level
            sell_price = config.center_price + spacing
            if config.direction in [GridDirection.BOTH, GridDirection.SHORT]:
                levels.append(GridLevel(
                    price=sell_price,
                    order_type="sell",
                    amount=config.amount_per_grid
                ))
                
        return levels
        
    async def _generate_fibonacci_grid(self, config: GridConfig, current_price: float) -> List[GridLevel]:
        """Generate Fibonacci-based grid levels"""
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        levels = []
        
        # Buy levels using Fibonacci retracements
        for fib in fib_levels:
            price = config.center_price * (1 - fib * config.price_range)
            if config.direction in [GridDirection.BOTH, GridDirection.LONG]:
                levels.append(GridLevel(
                    price=price,
                    order_type="buy",
                    amount=config.amount_per_grid
                ))
                
        # Sell levels using Fibonacci extensions
        for fib in fib_levels:
            price = config.center_price * (1 + fib * config.price_range)
            if config.direction in [GridDirection.BOTH, GridDirection.SHORT]:
                levels.append(GridLevel(
                    price=price,
                    order_type="sell",
                    amount=config.amount_per_grid
                ))
                
        return levels
        
    async def _generate_support_resistance_grid(self, config: GridConfig, current_price: float) -> List[GridLevel]:
        """Generate grid based on support/resistance levels"""
        # Fetch recent data for S/R analysis
        ohlcv = await asyncio.wait_for(
            asyncio.to_thread(
                self.exchange.fetch_ohlcv, config.symbol, "4h", None, 200
            ),
            timeout=30.0
        )
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Simple S/R detection using pivot points
        resistance_levels = self._find_resistance_levels(df, num_levels=5)
        support_levels = self._find_support_levels(df, num_levels=5)
        
        levels = []
        
        # Add buy orders near support levels
        if config.direction in [GridDirection.BOTH, GridDirection.LONG]:
            for support in support_levels:
                price = support * 1.001  # Slightly above support
                levels.append(GridLevel(
                    price=price,
                    order_type="buy",
                    amount=config.amount_per_grid
                ))
                
        # Add sell orders near resistance levels
        if config.direction in [GridDirection.BOTH, GridDirection.SHORT]:
            for resistance in resistance_levels:
                price = resistance * 0.999  # Slightly below resistance
                levels.append(GridLevel(
                    price=price,
                    order_type="sell",
                    amount=config.amount_per_grid
                ))
                
        # Store S/R levels for future reference
        self.support_resistance_levels[config.symbol] = (
            min(support_levels) if support_levels else current_price * 0.9,
            max(resistance_levels) if resistance_levels else current_price * 1.1
        )
        
        return levels
        
    def _find_resistance_levels(self, df: pd.DataFrame, num_levels: int = 5) -> List[float]:
        """Find resistance levels using pivot analysis"""
        highs = df['high'].rolling(window=10).max()
        resistance_candidates = []
        
        for i in range(10, len(df) - 10):
            # Resistance if price reached local high
            if df['high'].iloc[i] == highs.iloc[i]:
                resistance_candidates.append(df['high'].iloc[i])
                
        # Return top resistance levels
        if resistance_candidates:
            return sorted(set(resistance_candidates), reverse=True)[:num_levels]
        return []
        
    def _find_support_levels(self, df: pd.DataFrame, num_levels: int = 5) -> List[float]:
        """Find support levels using pivot analysis"""
        lows = df['low'].rolling(window=10).min()
        support_candidates = []
        
        for i in range(10, len(df) - 10):
            # Support if price reached local low
            if df['low'].iloc[i] == lows.iloc[i]:
                support_candidates.append(df['low'].iloc[i])
                
        # Return top support levels
        if support_candidates:
            return sorted(set(support_candidates))[:num_levels]
        return []
        
    async def _place_grid_orders(self, symbol: str) -> None:
        """Place all grid orders for a symbol"""
        if symbol not in self.active_grids:
            return
            
        config = self.grid_configs[symbol]
        levels = self.active_grids[symbol]
        
        for level in levels:
            if not level.filled:  # Only place orders for unfilled levels
                try:
                    order = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.exchange.create_limit_order,
                            symbol,
                            level.order_type,
                            level.amount,
                            level.price
                        ),
                        timeout=30.0
                    )
                    
                    level.order_id = order['id']
                    level.placed = True
                    
                    self.logger.info(
                        f"Grid {level.order_type} order placed: {symbol} @ {level.price} (ID: {order['id']})"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Failed to place grid order: {e}")
                    
    async def update_grid_status(self, symbol: str) -> None:
        """Update status of grid orders and handle fills"""
        if symbol not in self.active_grids:
            return
            
        # Check order status and update fills
        for level in self.active_grids[symbol]:
            if level.placed and level.order_id:
                try:
                    order_status = await asyncio.wait_for(
                        asyncio.to_thread(self.exchange.fetch_order, level.order_id, symbol),
                        timeout=10.0
                    )
                    
                    if order_status['status'] == 'closed':
                        level.filled = True
                        level.fill_time = datetime.now(timezone.utc)
                        
                        # Handle grid fill logic
                        await self._handle_grid_fill(symbol, level)
                        
                        self.logger.info(
                            f"Grid {level.order_type} order filled: {symbol} @ {level.price}"
                        )
                        
                except Exception as e:
                    self.logger.debug(f"Failed to check order status: {e}")
                    
    async def _handle_grid_fill(self, symbol: str, filled_level: GridLevel) -> None:
        """Handle grid order fill - place opposite order"""
        if symbol not in self.grid_configs:
            return
            
        config = self.grid_configs[symbol]
        
        # If buy order filled, place sell order higher
        if filled_level.order_type == "buy":
            # Find appropriate sell level above the fill price
            sell_price = filled_level.price * (1 + config.rebalance_threshold)
            
            try:
                order = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.exchange.create_limit_order,
                        symbol,
                        "sell",
                        filled_level.amount,
                        sell_price
                    ),
                    timeout=30.0
                )
                
                # Add new sell level to grid
                new_sell_level = GridLevel(
                    price=sell_price,
                    order_type="sell",
                    amount=filled_level.amount,
                    placed=True,
                    order_id=order['id']
                )
                
                self.active_grids[symbol].append(new_sell_level)
                self.logger.info(f"Grid sell order placed after buy fill @ {sell_price}")
                
            except Exception as e:
                self.logger.error(f"Failed to place sell order after grid fill: {e}")
                
        # If sell order filled, place buy order lower
        elif filled_level.order_type == "sell":
            # Find appropriate buy level below the fill price
            buy_price = filled_level.price * (1 - config.rebalance_threshold)
            
            try:
                order = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.exchange.create_limit_order,
                        symbol,
                        "buy",
                        filled_level.amount,
                        buy_price
                    ),
                    timeout=30.0
                )
                
                # Add new buy level to grid
                new_buy_level = GridLevel(
                    price=buy_price,
                    order_type="buy",
                    amount=filled_level.amount,
                    placed=True,
                    order_id=order['id']
                )
                
                self.active_grids[symbol].append(new_buy_level)
                self.logger.info(f"Grid buy order placed after sell fill @ {buy_price}")
                
            except Exception as e:
                self.logger.error(f"Failed to place buy order after grid fill: {e}")
                
    async def check_rebalancing(self, symbol: str) -> None:
        """Check if grid needs rebalancing based on price movement"""
        if symbol not in self.grid_configs:
            return
            
        config = self.grid_configs[symbol]
        
        if not config.enable_rebalancing:
            return
            
        # Get current price
        try:
            ticker = await asyncio.wait_for(
                asyncio.to_thread(self.exchange.fetch_ticker, symbol),
                timeout=10.0
            )
            current_price = ticker['last']
        except Exception:
            return
            
        # Check if price moved beyond rebalance threshold
        price_change = abs(current_price - config.center_price) / config.center_price
        
        if price_change > config.rebalance_threshold:
            self.logger.info(
                f"Rebalancing {symbol} due to price movement: {price_change:.2%} from center"
            )
            
            # Cancel existing orders and regenerate grid
            await self._cancel_all_grid_orders(symbol)
            config.center_price = current_price
            
            # Generate new grid levels
            new_levels = await self._generate_grid_levels(config, current_price)
            self.active_grids[symbol] = new_levels
            
            # Place new grid orders
            await self._place_grid_orders(symbol)
            
    async def _cancel_all_grid_orders(self, symbol: str) -> None:
        """Cancel all active grid orders for a symbol"""
        if symbol not in self.active_grids:
            return
            
        for level in self.active_grids[symbol]:
            if level.placed and level.order_id and not level.filled:
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(self.exchange.cancel_order, level.order_id, symbol),
                        timeout=10.0
                    )
                    level.placed = False
                    level.order_id = None
                    
                    self.logger.info(f"Cancelled grid order: {symbol} @ {level.price}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to cancel grid order: {e}")
                    
    async def get_grid_summary(self, symbol: str) -> Dict[str, Any]:
        """Get summary of grid trading performance"""
        if symbol not in self.grid_configs:
            return {"error": "No grid configured for this symbol"}
            
        config = self.grid_configs[symbol]
        levels = self.active_grids.get(symbol, [])
        
        # Calculate statistics
        total_orders = len(levels)
        filled_orders = sum(1 for level in levels if level.filled)
        active_orders = sum(1 for level in levels if level.is_active)
        
        # Calculate PnL
        realized_pnl = 0.0
        filled_count = 0
        
        for level in levels:
            if level.filled and level.order_type == "sell":
                # This is a simplified PnL calculation
                # In reality, you'd need to match buy/sell pairs
                filled_count += 1
                
        return {
            "symbol": symbol,
            "config": {
                "direction": config.direction.value,
                "grid_type": config.grid_type.value,
                "center_price": config.center_price,
                "num_grids": config.num_grids,
                "amount_per_grid": config.amount_per_grid,
                "leverage": config.leverage,
            },
            "status": {
                "total_orders": total_orders,
                "filled_orders": filled_orders,
                "active_orders": active_orders,
                "fill_rate": filled_orders / total_orders if total_orders > 0 else 0,
            },
            "performance": {
                "realized_pnl": realized_pnl,
                "trades_completed": filled_count // 2,  # Buy+sell pairs
            },
            "support_resistance": self.support_resistance_levels.get(symbol, (None, None)),
        }
        
    async def emergency_stop_grid(self, symbol: str) -> List[str]:
        """Emergency stop - cancel all orders and return status"""
        cancelled_orders = []
        
        if symbol in self.active_grids:
            await self._cancel_all_grid_orders(symbol)
            cancelled_orders = [level.order_id for level in self.active_grids[symbol] if level.order_id]
            
            # Clear grid
            del self.active_grids[symbol]
            del self.grid_configs[symbol]
            
            self.logger.warning(f"Emergency stop executed for {symbol}")
            
        return cancelled_orders