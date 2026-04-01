"""Grid Trading Bot for Reco-Trading
Provides automated grid trading with multiple modes (Classic, Infinity, Smart)
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import math

logger = logging.getLogger(__name__)


class GridMode(Enum):
    """Grid trading mode"""
    CLASSIC = "classic"
    INFINITY = "infinity"
    SMART = "smart"
    AUTO_BALANCE = "auto_balance"


class GridStatus(Enum):
    """Status of grid bot"""
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class GridLevel:
    """Individual grid level"""
    level_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    level_number: int = 0
    price: float = 0.0
    quantity: float = 0.0
    is_filled: bool = False
    filled_at: Optional[datetime] = None
    order_id: Optional[str] = None
    

@dataclass
class GridConfig:
    """Configuration for grid trading"""
    config_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Basic config
    symbol: str = ""
    mode: GridMode = GridMode.CLASSIC
    
    # Price range
    lower_price: float = 0.0
    upper_price: float = 0.0
    grid_count: int = 10
    
    # Position size
    total_investment: float = 1000.0
    order_quantity: float = 100.0
    
    # Advanced
    take_profit_percent: float = 2.0
    stop_loss_percent: float = 10.0
    use_trailing: bool = True
    trailing_distance: float = 1.0
    
    # Auto-balance (for infinity mode)
    auto_rebalance: bool = True
    rebalance_threshold: float = 5.0
    
    # Safety
    max_active_orders: int = 20
    cooldown_seconds: int = 5
    
    # Status
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class GridOrder:
    """Order placed by grid bot"""
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    config_id: str = ""
    grid_level: int = 0
    side: str = "buy"
    symbol: str = ""
    price: float = 0.0
    quantity: float = 0.0
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    filled_price: float = 0.0


@dataclass
class GridPosition:
    """Current position from grid trading"""
    position_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    config_id: str = ""
    symbol: str = ""
    
    # Quantities
    total_bought: float = 0.0
    total_sold: float = 0.0
    current_holding: float = 0.0
    
    # Costs
    total_invested: float = 0.0
    total_received: float = 0.0
    average_buy_price: float = 0.0
    
    # PnL
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    
    # Status
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class GridBot:
    """Grid Trading Bot"""
    
    def __init__(self, config: GridConfig):
        self.config = config
        self.levels: list[GridLevel] = []
        self.orders: list[GridOrder] = []
        self.position: Optional[GridPosition] = None
        self.status = GridStatus.PENDING
        self.start_price: float = 0.0
        self.current_price: float = 0.0
        self.total_profit: float = 0.0
        self.grid_spacing: float = 0.0
        
        self._initialize_grid()
        
    def _initialize_grid(self) -> None:
        """Initialize grid levels"""
        if self.config.lower_price >= self.config.upper_price:
            logger.error("Invalid price range for grid")
            self.status = GridStatus.ERROR
            return
            
        self.grid_spacing = (self.config.upper_price - self.config.lower_price) / self.config.grid_count
        self.start_price = self.config.lower_price
        
        for i in range(self.config.grid_count):
            level = GridLevel(
                level_number=i,
                price=self.config.lower_price + (i * self.grid_spacing),
                quantity=self.config.order_quantity
            )
            self.levels.append(level)
            
        self.position = GridPosition(
            config_id=self.config.config_id,
            symbol=self.config.symbol,
            total_invested=0.0,
            current_holding=0.0
        )
        
        logger.info(f"Grid initialized with {len(self.levels)} levels, spacing: {self.grid_spacing:.4f}")
        
    async def start(self, current_price: float) -> bool:
        """Start the grid bot"""
        self.current_price = current_price
        
        if current_price < self.config.lower_price or current_price > self.config.upper_price:
            logger.warning(f"Current price {current_price} is outside grid range")
            
        self.status = GridStatus.ACTIVE
        logger.info(f"Grid bot started for {self.config.symbol} at {current_price}")
        return True
    
    async def stop(self) -> None:
        """Stop the grid bot"""
        self.status = GridStatus.STOPPED
        logger.info(f"Grid bot stopped, total profit: {self.total_profit:.2f}")
        
    async def pause(self) -> None:
        """Pause the grid bot"""
        self.status = GridStatus.PAUSED
        logger.info("Grid bot paused")
        
    async def resume(self, current_price: float) -> bool:
        """Resume the grid bot"""
        if self.status != GridStatus.PAUSED:
            return False
            
        self.current_price = current_price
        self.status = GridStatus.ACTIVE
        logger.info("Grid bot resumed")
        return True
    
    async def check_price(self, current_price: float) -> list[GridOrder]:
        """Check current price and generate orders if needed"""
        self.current_price = current_price
        new_orders = []
        
        if self.status != GridStatus.ACTIVE:
            return new_orders
            
        active_orders = len([o for o in self.orders if o.status == "filled"])
        if active_orders >= self.config.max_active_orders:
            return new_orders
            
        # Find triggered levels
        for level in self.levels:
            if level.is_filled:
                continue
                
            # Buy trigger: price drops to level (classic) or below (infinity)
            if self.config.mode == GridMode.INFINITY:
                if current_price <= level.price:
                    order = await self._place_buy_order(level)
                    if order:
                        new_orders.append(order)
            else:
                if current_price <= level.price:
                    order = await self._place_buy_order(level)
                    if order:
                        new_orders.append(order)
                        
        return new_orders
    
    async def _place_buy_order(self, level: GridLevel) -> Optional[GridOrder]:
        """Place a buy order at a grid level"""
        if level.is_filled:
            return None
            
        order = GridOrder(
            config_id=self.config.config_id,
            grid_level=level.level_number,
            side="buy",
            symbol=self.config.symbol,
            price=level.price,
            quantity=level.quantity,
            status="pending"
        )
        
        self.orders.append(order)
        level.order_id = order.order_id
        
        logger.info(f"Buy order placed at level {level.level_number}: {level.price}")
        return order
    
    async def _place_sell_order(self, level: GridLevel) -> Optional[GridOrder]:
        """Place a sell order at a grid level"""
        if not level.is_filled:
            return None
            
        order = GridOrder(
            config_id=self.config.config_id,
            grid_level=level.level_number,
            side="sell",
            symbol=self.config.symbol,
            price=level.price + self.grid_spacing,
            quantity=level.quantity,
            status="pending"
        )
        
        self.orders.append(order)
        logger.info(f"Sell order placed at level {level.level_number}")
        return order
    
    async def on_order_filled(self, order: GridOrder, fill_price: float) -> list[GridOrder]:
        """Handle order filled event"""
        order.status = "filled"
        order.filled_at = datetime.now()
        order.filled_price = fill_price
        
        new_orders = []
        
        if order.side == "buy":
            # Mark level as filled
            level = next((l for l in self.levels if l.level_number == order.grid_level), None)
            if level:
                level.is_filled = True
                level.filled_at = datetime.now()
                
            # Update position
            if self.position:
                self.position.total_bought += order.quantity
                self.position.current_holding += order.quantity
                self.position.total_invested += order.quantity * fill_price
                self.position.average_buy_price = self.position.total_invested / self.position.total_bought
                
            # Place sell order at next level
            next_level_num = order.grid_level + 1
            if next_level_num < len(self.levels):
                next_level = self.levels[next_level_num]
                if not next_level.is_filled:
                    sell_order = await self._place_sell_order(next_level)
                    if sell_order:
                        new_orders.append(sell_order)
                        
        else:  # sell
            if self.position:
                self.position.total_sold += order.quantity
                self.position.current_holding -= order.quantity
                self.position.total_received += order.quantity * fill_price
                self.position.realized_pnl += (fill_price - self.position.average_buy_price) * order.quantity
                
        return new_orders
    
    async def check_take_profit(self, current_price: float) -> bool:
        """Check if take profit is triggered"""
        if not self.position or self.position.total_bought == 0:
            return False
            
        if self.config.take_profit_percent <= 0:
            return False
            
        current_profit_percent = ((current_price - self.position.average_buy_price) / 
                                   self.position.average_buy_price * 100)
        
        if current_profit_percent >= self.config.take_profit_percent:
            logger.info(f"Take profit triggered: {current_profit_percent:.2f}%")
            return True
            
        return False
    
    async def check_stop_loss(self, current_price: float) -> bool:
        """Check if stop loss is triggered"""
        if not self.position or self.position.total_bought == 0:
            return False
            
        if self.config.stop_loss_percent <= 0:
            return False
            
        current_loss_percent = ((self.position.average_buy_price - current_price) / 
                                 self.position.average_buy_price * 100)
        
        if current_loss_percent >= self.config.stop_loss_percent:
            logger.warning(f"Stop loss triggered: {current_loss_percent:.2f}%")
            return True
            
        return False
    
    def get_status(self) -> dict:
        """Get current status of grid bot"""
        filled_levels = len([l for l in self.levels if l.is_filled])
        active_orders = len([o for o in self.orders if o.status == "pending"])
        
        if self.position:
            self.position.unrealized_pnl = (
                self.position.current_holding * self.current_price - 
                (self.position.current_holding * self.position.average_buy_price)
            )
            self.position.total_pnl = self.position.realized_pnl + self.position.unrealized_pnl
            
        return {
            "config_id": self.config.config_id,
            "symbol": self.config.symbol,
            "mode": self.config.mode.value,
            "status": self.status.value,
            "current_price": self.current_price,
            "filled_levels": filled_levels,
            "total_levels": len(self.levels),
            "active_orders": active_orders,
            "total_profit": self.total_profit,
            "position": {
                "total_bought": self.position.total_bought if self.position else 0,
                "total_sold": self.position.total_sold if self.position else 0,
                "current_holding": self.position.current_holding if self.position else 0,
                "average_buy_price": self.position.average_buy_price if self.position else 0,
                "realized_pnl": self.position.realized_pnl if self.position else 0,
                "unrealized_pnl": self.position.unrealized_pnl if self.position else 0,
                "total_pnl": self.position.total_pnl if self.position else 0,
            }
        }


class GridBotManager:
    """Manages multiple grid bots"""
    
    def __init__(self):
        self.bots: dict[str, GridBot] = {}
        self.executor: Optional[Any] = None
        
    async def create_bot(self, config: GridConfig) -> str:
        """Create a new grid bot"""
        bot = GridBot(config)
        self.bots[config.config_id] = bot
        logger.info(f"Grid bot created: {config.config_id}")
        return config.config_id
    
    async def start_bot(self, config_id: str, current_price: float) -> bool:
        """Start a grid bot"""
        if config_id in self.bots:
            return await self.bots[config_id].start(current_price)
        return False
    
    async def stop_bot(self, config_id: str) -> None:
        """Stop a grid bot"""
        if config_id in self.bots:
            await self.bots[config_id].stop()
            
    async def delete_bot(self, config_id: str) -> bool:
        """Delete a grid bot"""
        if config_id in self.bots:
            await self.bots[config_id].stop()
            del self.bots[config_id]
            return True
        return False
    
    async def check_all_bots(self, prices: dict[str, float]) -> dict[str, list[GridOrder]]:
        """Check all bots for new orders"""
        results = {}
        
        for config_id, bot in self.bots.items():
            if bot.config.symbol in prices:
                orders = await bot.check_price(prices[bot.config.symbol])
                if orders:
                    results[config_id] = orders
                    
        return results
    
    def get_bot_status(self, config_id: str) -> Optional[dict]:
        """Get status of a specific bot"""
        if config_id in self.bots:
            return self.bots[config_id].get_status()
        return None
    
    def get_all_bots_status(self) -> list[dict]:
        """Get status of all bots"""
        return [bot.get_status() for bot in self.bots.values()]
    
    def get_active_bots_count(self) -> int:
        """Get count of active bots"""
        return len([b for b in self.bots.values() if b.status == GridStatus.ACTIVE])
    
    async def run_loop(self, get_prices_func) -> None:
        """Run the grid bot monitoring loop"""
        while True:
            try:
                prices = await get_prices_func()
                results = await self.check_all_bots(prices)
                
                # Execute orders
                for config_id, orders in results.items():
                    bot = self.bots[config_id]
                    for order in orders:
                        # Simulate order execution (in real app, call exchange)
                        await bot.on_order_filled(order, order.price)
                        
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Grid bot loop error: {e}")
                await asyncio.sleep(5)


# Global instance
_grid_bot_manager: Optional[GridBotManager] = None


def get_grid_bot_manager() -> GridBotManager:
    """Get or create the grid bot manager"""
    global _grid_bot_manager
    if _grid_bot_manager is None:
        _grid_bot_manager = GridBotManager()
    return _grid_bot_manager


__all__ = [
    "GridMode",
    "GridStatus",
    "GridLevel",
    "GridConfig",
    "GridOrder",
    "GridPosition",
    "GridBot",
    "GridBotManager",
    "get_grid_bot_manager",
]