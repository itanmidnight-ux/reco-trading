from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from reco_trading.agents.base_agent import (
    BaseLLMAgent, AgentConfig, AgentRole, AgentCapability, TradingDecision
)


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float | None = None
    stop_price: float | None = None
    order_id: str = ""
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: datetime | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ExecutionResult:
    success: bool
    order: Order | None = None
    error: str | None = None
    execution_time: float = 0.0
    slippage: float = 0.0
    metadata: dict = field(default_factory=dict)


class ExecutorAgent(BaseLLMAgent):
    def __init__(self, config: AgentConfig | None = None):
        if config is None:
            config = AgentConfig(
                name="executor",
                role=AgentRole.EXECUTOR,
                capabilities=[AgentCapability.TRADE_EXECUTION],
                model="llama3",
                temperature=0.1,
                max_tokens=256
            )
        super().__init__(config)
        
        self._exchange_client = None
        self._pending_orders: dict[str, Order] = {}
        self._execution_history: list[ExecutionResult] = []
        
        self._default_slippage = 0.001
        self._max_retries = 3
        self._retry_delay = 1.0

    def set_exchange_client(self, exchange_client) -> None:
        self._exchange_client = exchange_client
        self.logger.info("Exchange client connected to Executor Agent")

    async def process(self, input_data: dict) -> dict:
        decision = input_data.get("decision")
        
        if not decision:
            return {
                "success": False,
                "error": "No trading decision provided"
            }
        
        execution_result = await self._execute_decision(decision)
        
        self._execution_history.append(execution_result)
        
        return {
            "agent": self.config.name,
            "result": execution_result,
            "order": execution_result.order
        }

    async def _execute_decision(self, decision: TradingDecision) -> ExecutionResult:
        import time
        start_time = time.time()
        
        action = decision.action.upper()
        
        if action == "HOLD":
            return ExecutionResult(
                success=True,
                error="No execution needed - HOLD signal",
                execution_time=time.time() - start_time
            )
        
        symbol = decision.metadata.get("symbol", "UNKNOWN")
        quantity = decision.metadata.get("quantity", 0.001)
        price = decision.metadata.get("price")
        
        if action == "BUY":
            return await self._execute_order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=quantity,
                price=price,
                execution_time_start=start_time
            )
        elif action == "SELL":
            return await self._execute_order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=quantity,
                price=price,
                execution_time_start=start_time
            )
        
        return ExecutionResult(
            success=False,
            error=f"Unknown action: {action}",
            execution_time=time.time() - start_time
        )

    async def _execute_order(self, symbol: str, side: OrderSide, 
                           quantity: float, price: float | None,
                           execution_time_start: float) -> ExecutionResult:
        import time
        
        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET if price is None else OrderType.LIMIT,
            quantity=quantity,
            price=price
        )
        
        self._pending_orders[order.order_id] = order
        
        for attempt in range(self._max_retries):
            try:
                if self._exchange_client:
                    result = await self._execute_via_exchange(order)
                else:
                    result = await self._simulate_execution(order)
                
                execution_time = time.time() - execution_time_start
                
                if result.get("success"):
                    order.status = "filled"
                    order.filled_at = datetime.now()
                    
                    slippage = self._calculate_slippage(
                        order.price or 0,
                        result.get("fill_price", order.price or 0)
                    )
                    
                    return ExecutionResult(
                        success=True,
                        order=order,
                        execution_time=execution_time,
                        slippage=slippage,
                        metadata=result
                    )
                else:
                    self.logger.warning(f"Execution attempt {attempt + 1} failed: {result.get('error')}")
                    
            except Exception as e:
                self.logger.error(f"Execution error: {e}")
                await asyncio.sleep(self._retry_delay)
        
        order.status = "failed"
        
        return ExecutionResult(
            success=False,
            order=order,
            error="Max retries exceeded",
            execution_time=time.time() - execution_time_start
        )

    async def _execute_via_exchange(self, order: Order) -> dict:
        try:
            if order.side == OrderSide.BUY:
                order_type = "buy"
            else:
                order_type = "sell"
            
            result = await self._exchange_client.create_order(
                symbol=order.symbol,
                type=order.order_type.value,
                side=order_type,
                quantity=order.quantity,
                price=order.price
            )
            
            return {
                "success": True,
                "order_id": result.get("id", ""),
                "fill_price": result.get("price", order.price),
                "fill_qty": result.get("executed_qty", order.quantity)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _simulate_execution(self, order: Order) -> dict:
        await asyncio.sleep(0.1)
        
        return {
            "success": True,
            "order_id": f"sim_{datetime.now().timestamp()}",
            "fill_price": order.price or 50000.0,
            "fill_qty": order.quantity
        }

    def _calculate_slippage(self, expected_price: float, actual_price: float) -> float:
        if expected_price == 0:
            return 0.0
        
        return abs(actual_price - expected_price) / expected_price

    async def cancel_order(self, order_id: str) -> bool:
        if order_id in self._pending_orders:
            order = self._pending_orders[order_id]
            
            if self._exchange_client:
                try:
                    await self._exchange_client.cancel_order(order_id, order.symbol)
                    order.status = "cancelled"
                    return True
                except Exception as e:
                    self.logger.error(f"Cancel failed: {e}")
                    return False
            
            order.status = "cancelled"
            return True
        
        return False

    async def get_order_status(self, order_id: str) -> Order | None:
        if order_id in self._pending_orders:
            return self._pending_orders[order_id]
        
        if self._exchange_client:
            try:
                result = await self._exchange_client.fetch_order(order_id)
                return result
            except:
                pass
        
        return None

    def get_pending_orders(self) -> list[Order]:
        return list(self._pending_orders.values())

    def get_execution_history(self, limit: int = 50) -> list[ExecutionResult]:
        return self._execution_history[-limit:]

    def get_execution_stats(self) -> dict:
        if not self._execution_history:
            return {"total_executions": 0}
        
        successful = sum(1 for e in self._execution_history if e.success)
        failed = len(self._execution_history) - successful
        
        avg_slippage = sum(e.slippage for e in self._execution_history) / len(self._execution_history)
        avg_time = sum(e.execution_time for e in self._execution_history) / len(self._execution_history)
        
        return {
            "total_executions": len(self._execution_history),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(self._execution_history),
            "avg_slippage": avg_slippage,
            "avg_execution_time": avg_time,
            "pending_orders": len(self._pending_orders)
        }


class OrderManager:
    def __init__(self):
        self._orders: dict[str, Order] = {}
        self._order_counter = 0

    def create_order(self, symbol: str, side: OrderSide, 
                    order_type: OrderType, quantity: float,
                    price: float | None = None) -> Order:
        self._order_counter += 1
        
        order = Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            order_id=f"ord_{self._order_counter}_{int(datetime.now().timestamp())}"
        )
        
        self._orders[order.order_id] = order
        
        return order

    def get_order(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)

    def update_order_status(self, order_id: str, status: str) -> bool:
        if order_id in self._orders:
            self._orders[order_id].status = status
            return True
        return False

    def get_active_orders(self) -> list[Order]:
        return [o for o in self._orders.values() if o.status in ["pending", "open"]]

    def get_filled_orders(self, limit: int = 100) -> list[Order]:
        filled = [o for o in self._orders.values() if o.status == "filled"]
        return filled[-limit:]