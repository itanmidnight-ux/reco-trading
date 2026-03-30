from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SafetyOrder:
    """Safety order configuration."""
    order_number: int
    volume: float
    price_offset: float
    triggered: bool = False
    order_id: str | None = None


@dataclass
class DCAConfig:
    """Configuration for Dollar Cost Averaging (DCA)."""
    enabled: bool = True
    max_safety_orders: int = 5
    max_grid_levels: int = 10
    safety_order_step: float = 0.02
    safety_order_volume_scale: float = 1.1
    price_deviation_threshold: float = 0.03
    enable_partial_take_profit: bool = True
    partial_take_profit_ratio: float = 0.5
    base_order_size: float = 10.0
    safety_order_size: float = 10.0
    dca_learn_mode: bool = False
    dca_learn_collection_size: int = 100


class DCAManager:
    """
    Enhanced DCA (Dollar Cost Averaging) Manager.
    Handles safety orders, partial take profits, and grid trading.
    """

    def __init__(self, config: DCAConfig | None = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or DCAConfig()
        self.active_safety_orders: dict[int, list[SafetyOrder]] = {}
        self.completed_dca_cycles: int = 0
        self.total_safety_orders_filled: int = 0

    def initialize_dca(
        self,
        trade_id: int,
        entry_price: float,
        base_order_size: float,
    ) -> list[SafetyOrder]:
        """Initialize DCA safety orders for a new trade."""
        
        if not self.config.enabled:
            return []
        
        safety_orders = []
        
        for i in range(1, self.config.max_safety_orders + 1):
            offset = self.config.safety_order_step * i
            
            volume_scale = self.config.safety_order_volume_scale ** (i - 1)
            volume = base_order_size * volume_scale
            
            safety_order = SafetyOrder(
                order_number=i,
                volume=volume,
                price_offset=offset,
            )
            safety_orders.append(safety_order)
        
        self.active_safety_orders[trade_id] = safety_orders
        
        self.logger.info(
            f"Initialized DCA for trade {trade_id}: "
            f"{len(safety_orders)} safety orders, entry: {entry_price}"
        )
        
        return safety_orders

    def check_safety_order_trigger(
        self,
        trade_id: int,
        current_price: float,
        entry_price: float,
        side: str = "BUY",
    ) -> SafetyOrder | None:
        """Check if a safety order should be triggered."""
        
        if trade_id not in self.active_safety_orders:
            return None
        
        safety_orders = self.active_safety_orders[trade_id]
        
        for safety_order in safety_orders:
            if safety_order.triggered:
                continue
            
            trigger_price = self._calculate_trigger_price(
                entry_price,
                safety_order.price_offset,
                side,
            )
            
            if self._should_trigger(current_price, trigger_price, side):
                safety_order.triggered = True
                self.total_safety_orders_filled += 1
                
                self.logger.info(
                    f"Safety order {safety_order.order_number} triggered for trade {trade_id} "
                    f"at {current_price} (offset: {safety_order.price_offset:.2%})"
                )
                
                return safety_order
        
        return None

    def _calculate_trigger_price(
        self,
        entry_price: float,
        price_offset: float,
        side: str,
    ) -> float:
        """Calculate trigger price for safety order."""
        
        if side.upper() == "BUY":
            return entry_price * (1 - price_offset)
        else:
            return entry_price * (1 + price_offset)

    def _should_trigger(self, current_price: float, trigger_price: float, side: str) -> bool:
        """Determine if safety order should be triggered."""
        
        if side.upper() == "BUY":
            return current_price <= trigger_price
        else:
            return current_price >= trigger_price

    def calculate_dca_entry(
        self,
        trade_id: int,
        filled_orders: list[dict],
    ) -> float | None:
        """Calculate average entry price after DCA."""
        
        if not filled_orders:
            return None
        
        total_cost = sum(o.get("filled", 0) * o.get("price", 0) for o in filled_orders)
        total_quantity = sum(o.get("filled", 0) for o in filled_orders)
        
        if total_quantity > 0:
            avg_price = total_cost / total_quantity
            self.logger.info(f"DCA average entry for trade {trade_id}: {avg_price}")
            return avg_price
        
        return None

    def check_partial_take_profit(
        self,
        trade_id: int,
        current_price: float,
        entry_price: float,
        side: str = "BUY",
    ) -> bool:
        """Check if partial take profit should be executed."""
        
        if not self.config.enable_partial_take_profit:
            return False
        
        profit_ratio = (current_price - entry_price) / entry_price if side.upper() == "BUY" else (entry_price - current_price) / entry_price
        
        if profit_ratio >= self.config.partial_take_profit_ratio:
            self.logger.info(
                f"Partial take profit triggered for trade {trade_id}: "
                f"profit {profit_ratio:.2%} >= {self.config.partial_take_profit_ratio:.2%}"
            )
            return True
        
        return False

    def calculate_remaining_safety_orders(self, trade_id: int) -> int:
        """Calculate remaining unfilled safety orders."""
        
        if trade_id not in self.active_safety_orders:
            return 0
        
        return sum(
            1 for so in self.active_safety_orders[trade_id]
            if not so.triggered
        )

    def close_dca(self, trade_id: int) -> dict:
        """Close DCA tracking for a trade."""
        
        if trade_id in self.active_safety_orders:
            triggered_count = sum(
                1 for so in self.active_safety_orders[trade_id]
                if so.triggered
            )
            
            info = {
                "trade_id": trade_id,
                "total_safety_orders": len(self.active_safety_orders[trade_id]),
                "triggered_count": triggered_count,
            }
            
            del self.active_safety_orders[trade_id]
            self.completed_dca_cycles += 1
            
            return info
        
        return {"trade_id": trade_id, "total_safety_orders": 0, "triggered_count": 0}

    def get_dca_status(self, trade_id: int) -> dict | None:
        """Get DCA status for a trade."""
        
        if trade_id not in self.active_safety_orders:
            return None
        
        safety_orders = self.active_safety_orders[trade_id]
        
        return {
            "trade_id": trade_id,
            "total_safety_orders": len(safety_orders),
            "triggered_count": sum(1 for so in safety_orders if so.triggered),
            "remaining_count": sum(1 for so in safety_orders if not so.triggered),
            "safety_orders": [
                {
                    "order_number": so.order_number,
                    "volume": so.volume,
                    "price_offset": so.price_offset,
                    "triggered": so.triggered,
                }
                for so in safety_orders
            ],
        }

    def get_stats(self) -> dict:
        """Get DCA statistics."""
        
        return {
            "enabled": self.config.enabled,
            "max_safety_orders": self.config.max_safety_orders,
            "safety_order_step": self.config.safety_order_step,
            "active_dca_trades": len(self.active_safety_orders),
            "completed_dca_cycles": self.completed_dca_cycles,
            "total_safety_orders_filled": self.total_safety_orders_filled,
        }

    def learn_from_trade(self, trade_data: dict) -> None:
        """Learn from completed trade to improve DCA parameters."""
        
        if not self.config.dca_learn_mode:
            return
        
        self.logger.info(f"Learning from trade: {trade_data.get('trade_id')}")


def calculate_dca_profit(
    entry_prices: list[float],
    quantities: list[float],
    exit_price: float,
    side: str = "BUY",
    fees: float = 0.001,
) -> dict:
    """Calculate profit from DCA strategy."""
    
    if len(entry_prices) != len(quantities):
        raise ValueError("Entry prices and quantities must have same length")
    
    total_cost = sum(p * q for p, q in zip(entry_prices, quantities))
    total_quantity = sum(quantities)
    
    if total_quantity == 0:
        return {"profit": 0, "profit_percent": 0, "fees": 0}
    
    avg_entry = total_cost / total_quantity
    exit_value = exit_price * total_quantity
    total_fees = total_cost * fees + exit_value * fees
    
    if side.upper() == "BUY":
        profit = exit_value - total_cost - total_fees
    else:
        profit = total_cost - exit_value - total_fees
    
    profit_percent = (profit / total_cost) * 100 if total_cost > 0 else 0
    
    return {
        "profit": profit,
        "profit_percent": profit_percent,
        "fees": total_fees,
        "avg_entry": avg_entry,
        "exit_price": exit_price,
        "total_quantity": total_quantity,
    }
