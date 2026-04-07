"""
Advanced Execution Engine Module.

This module provides sophisticated order execution with:
- Smart order routing
- Slippage minimization
- Latency optimization
- Order splitting
- Fill verification
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"


@dataclass
class ExecutionConfig:
    """Configuration for execution engine."""
    max_slippage_percent: float = 0.3  # Maximum allowed slippage (%)
    max_spread_percent: float = 0.2  # Maximum spread to enter (%)
    order_timeout_seconds: float = 5.0  # Timeout for order submission
    fill_timeout_seconds: float = 10.0  # Timeout for order fill verification
    split_threshold_usdt: float = 500.0  # Split orders above this size
    max_split_parts: int = 5  # Maximum number of order splits
    retry_attempts: int = 3  # Number of retry attempts
    retry_delay_seconds: float = 1.0  # Delay between retries
    use_websocket: bool = True  # Use WebSocket for faster execution
    verify_fills: bool = True  # Verify order fills
    min_fill_percent: float = 95.0  # Minimum fill percentage to consider success


@dataclass
class OrderRequest:
    """Order request to be executed."""
    symbol: str
    side: str  # BUY or SELL
    quantity: float
    price: float | None = None  # None for market orders
    order_type: OrderType = OrderType.MARKET
    client_order_id: str | None = None
    stop_price: float | None = None  # For stop orders
    take_profit_price: float | None = None
    time_in_force: str = "GTC"  # GTC, IOC, FOK
    reduce_only: bool = False


@dataclass
class OrderResult:
    """Result of order execution."""
    order_id: str | None
    client_order_id: str | None
    symbol: str
    side: str
    requested_quantity: float
    filled_quantity: float
    average_price: float
    status: OrderStatus
    timestamp: datetime
    execution_time_ms: float
    slippage_percent: float
    spread_percent: float
    split_parts: int = 1
    error_message: str | None = None
    fees: float = 0.0


@dataclass
class ExecutionStats:
    """Execution statistics."""
    total_orders: int = 0
    successful_orders: int = 0
    failed_orders: int = 0
    total_slippage: float = 0.0
    avg_slippage: float = 0.0
    total_volume_usdt: float = 0.0
    avg_latency_ms: float = 0.0
    total_fees: float = 0.0


class SmartOrderRouter:
    """Smart order routing with slippage minimization."""
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self._last_spreads: dict[str, list[float]] = {}
        self._last_prices: dict[str, tuple[float, float, float]] = {}  # (bid, ask, timestamp)
    
    def should_execute(
        self,
        symbol: str,
        side: str,
        quantity: float,
        current_price: float,
        bid: float | None = None,
        ask: float | None = None,
    ) -> tuple[bool, str]:
        """
        Determine if order should be executed based on spread and conditions.
        
        Returns:
            (should_execute, reason)
        """
        if bid is None or ask is None:
            return True, "no_spread_data"
        
        spread = abs(ask - bid)
        spread_percent = (spread / current_price) * 100 if current_price > 0 else 0
        
        # Check spread threshold
        if spread_percent > self.config.max_spread_percent:
            return False, f"spread_too_high: {spread_percent:.3f}% > {self.config.max_spread_percent}%"
        
        # Check for usual market conditions
        if side.upper() == "BUY":
            # For buy orders, check if ask is reasonable
            if ask > current_price * 1.005:  # Ask is 0.5% above last price
                return False, f"ask_too_high: {ask:.8f} > {current_price * 1.005:.8f}"
        else:
            # For sell orders, check if bid is reasonable
            if bid < current_price * 0.995:  # Bid is 0.5% below last price
                return False, f"bid_too_low: {bid:.8f} < {current_price * 0.995:.8f}"
        
        return True, "ok"
    
    def calculate_order_split(
        self,
        quantity: float,
        price: float,
        quantity_usdt: float,
    ) -> list[float]:
        """
        Calculate how to split large orders to minimize slippage.
        
        Returns:
            List of quantities for each split order
        """
        if quantity_usdt <= self.config.split_threshold_usdt:
            return [quantity]
        
        # Calculate optimal split
        value_ratio = quantity_usdt / self.config.split_threshold_usdt
        ideal_parts = int(value_ratio) + 1
        num_parts = min(ideal_parts, self.config.max_split_parts)
        
        # Distribute quantity across parts
        base_qty = quantity / num_parts
        
        # Use slightly decreasing quantities for later parts (front-load execution)
        parts = []
        remaining = quantity
        for i in range(num_parts - 1):
            # Front-load: first orders are slightly larger
            part_qty = base_qty * (1.0 - i * 0.02)  # 2% decrease per part
            parts.append(part_qty)
            remaining -= part_qty
        
        parts.append(remaining)  # Final part
        
        return [max(q, 0.0) for q in parts if q > 0]


class OrderFillVerifier:
    """Verifies order fills and handles partial fills."""
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self._pending_orders: dict[str, OrderRequest] = {}
        self._order_timestamps: dict[str, float] = {}
    
    async def verify_fill(
        self,
        order_id: str,
        expected_quantity: float,
        client: Any,
        timeout: float | None = None,
    ) -> tuple[bool, float, str]:
        """
        Verify that an order was filled.
        
        Returns:
            (success, filled_quantity, status_message)
        """
        timeout = timeout or self.config.fill_timeout_seconds
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Fetch order status from exchange
                order_info = await self._fetch_order_status(order_id, client)
                
                if order_info is None:
                    await asyncio.sleep(0.1)
                    continue
                
                status = order_info.get("status", "")
                filled_qty = float(order_info.get("filled", 0) or order_info.get("filledQty", 0) or 0)
                
                if status in ("closed", "filled"):
                    fill_percent = (filled_qty / expected_quantity * 100) if expected_quantity > 0 else 100
                    if fill_percent >= self.config.min_fill_percent:
                        return True, filled_qty, f"filled_{fill_percent:.1f}%"
                    else:
                        return False, filled_qty, f"partial_fill_{fill_percent:.1f}%"
                elif status == "cancelled":
                    return False, filled_qty, "cancelled"
                elif status == "rejected":
                    return False, 0.0, "rejected"
                elif status == "expired":
                    return False, filled_qty, "expired"
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Order verification error: {e}")
                await asyncio.sleep(0.2)
        
        return False, 0.0, "timeout"
    
    async def _fetch_order_status(self, order_id: str, client: Any) -> dict | None:
        """Fetch order status from exchange."""
        try:
            if hasattr(client, "fetch_order"):
                return await client.fetch_order(order_id)
            elif hasattr(client, "get_order"):
                return await client.get_order(order_id)
        except Exception as e:
            logger.debug(f"Failed to fetch order {order_id}: {e}")
        return None


class LatencyOptimizer:
    """Optimizes execution latency."""
    
    def __init__(self):
        self._latency_history: list[float] = []
        self._max_history = 100
    
    def record_latency(self, latency_ms: float) -> None:
        """Record execution latency."""
        self._latency_history.append(latency_ms)
        if len(self._latency_history) > self._max_history:
            self._latency_history = self._latency_history[-self._max_history:]
    
    def get_avg_latency(self) -> float:
        """Get average execution latency."""
        if not self._latency_history:
            return 0.0
        return sum(self._latency_history) / len(self._latency_history)
    
    def get_p95_latency(self) -> float:
        """Get 95th percentile latency."""
        if not self._latency_history:
            return 0.0
        sorted_latency = sorted(self._latency_history)
        idx = int(len(sorted_latency) * 0.95)
        return sorted_latency[min(idx, len(sorted_latency) - 1)]


class SmartExecutor:
    """
    Main execution engine that coordinates smart order routing,
    slippage minimization, and latency optimization.
    """
    
    def __init__(self, config: ExecutionConfig | None = None):
        self.config = config or ExecutionConfig()
        self.router = SmartOrderRouter(self.config)
        self.verifier = OrderFillVerifier(self.config)
        self.latency_optimizer = LatencyOptimizer()
        self.stats = ExecutionStats()
        self._lock = asyncio.Lock()
    
    async def execute(
        self,
        client: Any,
        request: OrderRequest,
        bid: float | None = None,
        ask: float | None = None,
        current_price: float | None = None,
    ) -> OrderResult:
        """
        Execute an order with smart routing and slippage minimization.
        
        Args:
            client: Exchange client
            request: Order request
            bid: Current bid price
            ask: Current ask price
            current_price: Current market price
        
        Returns:
            OrderResult with execution details
        """
        start_time = time.time()
        current_price = current_price or request.price or ask or bid or 0.0
        
        # Determine current price from market data
        if current_price is None or current_price == 0:
            if bid and ask:
                current_price = (bid + ask) / 2
            elif bid:
                current_price = bid
            elif ask:
                current_price = ask
        
        # Check if should execute based on spread
        should_exec, reason = self.router.should_execute(
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            current_price=current_price,
            bid=bid,
            ask=ask,
        )
        
        if not should_exec:
            return OrderResult(
                order_id=None,
                client_order_id=request.client_order_id,
                symbol=request.symbol,
                side=request.side,
                requested_quantity=request.quantity,
                filled_quantity=0.0,
                average_price=0.0,
                status=OrderStatus.REJECTED,
                timestamp=datetime.now(timezone.utc),
                execution_time_ms=0.0,
                slippage_percent=0.0,
                spread_percent=0.0,
                split_parts=0,
                error_message=reason,
            )
        
        # Calculate order value and split if necessary
        order_value = request.quantity * current_price
        
        if request.order_type == OrderType.MARKET and order_value > self.config.split_threshold_usdt:
            # Split large market orders
            return await self._execute_split_order(
                client=client,
                request=request,
                current_price=current_price,
                bid=bid,
                ask=ask,
            )
        else:
            # Execute single order
            return await self._execute_single_order(
                client=client,
                request=request,
                current_price=current_price,
                bid=bid,
                ask=ask,
            )
    
    async def _execute_single_order(
        self,
        client: Any,
        request: OrderRequest,
        current_price: float,
        bid: float | None,
        ask: float | None,
    ) -> OrderResult:
        """Execute a single order."""
        start_time = time.time()
        
        best_price = current_price
        if request.order_type == OrderType.LIMIT and request.price:
            best_price = request.price
        elif request.side.upper() == "BUY" and ask:
            best_price = ask
        elif request.side.upper() == "SELL" and bid:
            best_price = bid
        
        for attempt in range(self.config.retry_attempts):
            try:
                order = await self._submit_order(
                    client=client,
                    request=request,
                    price=best_price,
                )
                
                if order is None:
                    continue
                
                order_id = order.get("id") or order.get("orderId")
                status = order.get("status", "unknown")
                
                # Safely extract filled quantity
                filled_raw = order.get("filled", 0) or order.get("filledQty", 0) or 0
                if isinstance(filled_raw, dict):
                    filled_qty = float(filled_raw.get("filled", 0) or filled_raw.get("cumQty", 0) or 0)
                else:
                    try:
                        filled_qty = float(filled_raw)
                    except (TypeError, ValueError):
                        filled_qty = 0.0
                
                # Safely extract average price
                avg_raw = order.get("average", 0) or order.get("avgPrice", 0) or best_price
                if isinstance(avg_raw, dict):
                    avg_price = float(avg_raw.get("avgPrice", 0) or avg_raw.get("price", 0) or best_price)
                else:
                    try:
                        avg_price = float(avg_raw)
                    except (TypeError, ValueError):
                        avg_price = float(best_price)
                
                # Safely extract fees
                fee_raw = order.get("fee", 0) or 0
                if isinstance(fee_raw, dict):
                    fee_val = float(fee_raw.get("cost", 0) or fee_raw.get("fee", 0) or 0)
                else:
                    try:
                        fee_val = float(fee_raw)
                    except (TypeError, ValueError):
                        fee_val = 0.0
                
                # Verify fill for market orders
                if request.order_type == OrderType.MARKET and self.config.verify_fills:
                    verified, verified_qty, status_msg = await self.verifier.verify_fill(
                        order_id=order_id,
                        expected_quantity=request.quantity,
                        client=client,
                    )
                    if verified:
                        filled_qty = max(filled_qty, verified_qty)
                
                execution_time_ms = (time.time() - start_time) * 1000
                
                # Calculate slippage
                slippage = 0.0
                if current_price > 0 and avg_price > 0:
                    if request.side.upper() == "BUY":
                        slippage = ((avg_price - current_price) / current_price) * 100
                    else:
                        slippage = ((current_price - avg_price) / current_price) * 100
                
                # Calculate spread
                spread = 0.0
                if bid and ask and current_price > 0:
                    spread = ((ask - bid) / current_price) * 100
                
                # Update stats
                self._update_stats(
                    success=filled_qty > 0,
                    slippage=slippage,
                    volume=request.quantity * avg_price,
                    latency_ms=execution_time_ms,
                    fees=fee_val,
                )
                
                self.latency_optimizer.record_latency(execution_time_ms)
                
                return OrderResult(
                    order_id=str(order_id),
                    client_order_id=request.client_order_id,
                    symbol=request.symbol,
                    side=request.side,
                    requested_quantity=request.quantity,
                    filled_quantity=filled_qty,
                    average_price=avg_price,
                    status=OrderStatus.FILLED if filled_qty >= request.quantity else OrderStatus.PARTIALLY_FILLED,
                    timestamp=datetime.now(timezone.utc),
                    execution_time_ms=execution_time_ms,
                    slippage_percent=slippage,
                    spread_percent=spread,
                    split_parts=1,
                    fees=fee_val,
                )
                
            except Exception as e:
                logger.warning(f"Order attempt {attempt + 1} failed: {e}")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay_seconds)
                continue
        
        # All retries failed
        execution_time_ms = (time.time() - start_time) * 1000
        self._update_stats(success=False, slippage=0.0, volume=0.0, latency_ms=execution_time_ms, fees=0.0)
        
        return OrderResult(
            order_id=None,
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            side=request.side,
            requested_quantity=request.quantity,
            filled_quantity=0.0,
            average_price=0.0,
            status=OrderStatus.FAILED,
            timestamp=datetime.now(timezone.utc),
            execution_time_ms=execution_time_ms,
            slippage_percent=0.0,
            spread_percent=0.0,
            split_parts=0,
            error_message="execution_failed_after_retries",
        )
    
    async def _execute_split_order(
        self,
        client: Any,
        request: OrderRequest,
        current_price: float,
        bid: float | None,
        ask: float | None,
    ) -> OrderResult:
        """Execute a large order split into multiple smaller orders."""
        start_time = time.time()
        
        # Calculate split
        order_value = request.quantity * current_price
        split_quantities = self.router.calculate_order_split(
            quantity=request.quantity,
            price=current_price,
            quantity_usdt=order_value,
        )
        
        total_filled = 0.0
        total_value = 0.0
        total_fees = 0.0
        successful_parts = 0
        
        for i, qty in enumerate(split_quantities):
            # Small delay between split orders to avoid rate limiting
            if i > 0:
                await asyncio.sleep(0.1)
            
            split_request = OrderRequest(
                symbol=request.symbol,
                side=request.side,
                quantity=qty,
                price=request.price,
                order_type=request.order_type,
                client_order_id=f"{request.client_order_id}_split_{i}" if request.client_order_id else None,
            )
            
            result = await self._execute_single_order(
                client=client,
                request=split_request,
                current_price=current_price,
                bid=bid,
                ask=ask,
            )
            
            if result.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
                total_filled += result.filled_quantity
                total_value += result.filled_quantity * result.average_price
                total_fees += result.fees
                successful_parts += 1
        
        execution_time_ms = (time.time() - start_time) * 1000
        avg_price = total_value / total_filled if total_filled > 0 else 0.0
        
        # Calculate slippage
        slippage = 0.0
        if current_price > 0 and avg_price > 0:
            if request.side.upper() == "BUY":
                slippage = ((avg_price - current_price) / current_price) * 100
            else:
                slippage = ((current_price - avg_price) / current_price) * 100
        
        spread = 0.0
        if bid and ask and current_price > 0:
            spread = ((ask - bid) / current_price) * 100
        
        status = OrderStatus.FILLED if total_filled >= request.quantity * 0.95 else OrderStatus.PARTIALLY_FILLED
        if total_filled == 0:
            status = OrderStatus.FAILED
        
        self.latency_optimizer.record_latency(execution_time_ms)
        
        return OrderResult(
            order_id=f"split_{request.client_order_id}" if request.client_order_id else None,
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            side=request.side,
            requested_quantity=request.quantity,
            filled_quantity=total_filled,
            average_price=avg_price,
            status=status,
            timestamp=datetime.now(timezone.utc),
            execution_time_ms=execution_time_ms,
            slippage_percent=slippage,
            spread_percent=spread,
            split_parts=len(split_quantities),
            fees=total_fees,
            error_message=None if total_filled > 0 else "split_execution_failed",
        )
    
    async def _submit_order(
        self,
        client: Any,
        request: OrderRequest,
        price: float,
    ) -> dict | None:
        """Submit order to exchange."""
        try:
            side = request.side.lower()
            
            if request.order_type == OrderType.MARKET:
                # Market order
                if hasattr(client, "create_market_order"):
                    return await client.create_market_order(
                        symbol=request.symbol,
                        side=side,
                        amount=request.quantity,
                        client_order_id=request.client_order_id,
                    )
                elif hasattr(client, "create_order"):
                    return await client.create_order(
                        symbol=request.symbol,
                        type="market",
                        side=side,
                        amount=request.quantity,
                        client_order_id=request.client_order_id,
                    )
            else:
                # Limit order
                if hasattr(client, "create_limit_order"):
                    return await client.create_limit_order(
                        symbol=request.symbol,
                        side=side,
                        amount=request.quantity,
                        price=price,
                        client_order_id=request.client_order_id,
                    )
                elif hasattr(client, "create_order"):
                    return await client.create_order(
                        symbol=request.symbol,
                        type="limit",
                        side=side,
                        amount=request.quantity,
                        price=price,
                        client_order_id=request.client_order_id,
                    )
        except Exception as e:
            logger.error(f"Order submission failed: {e}")
            return None
        
        return None
    
    def _update_stats(
        self,
        success: bool,
        slippage: float,
        volume: float,
        latency_ms: float,
        fees: float,
    ) -> None:
        """Update execution statistics."""
        self.stats.total_orders += 1
        if success:
            self.stats.successful_orders += 1
        else:
            self.stats.failed_orders += 1
        
        self.stats.total_slippage += abs(slippage)
        self.stats.avg_slippage = self.stats.total_slippage / self.stats.total_orders
        self.stats.total_volume_usdt += volume
        self.stats.total_fees += fees
        
        # Update average latency
        current_avg = self.stats.avg_latency_ms
        n = self.stats.total_orders
        self.stats.avg_latency_ms = current_avg + (latency_ms - current_avg) / n
    
    def get_stats(self) -> ExecutionStats:
        """Get execution statistics."""
        return self.stats


__all__ = [
    "ExecutionConfig",
    "OrderRequest",
    "OrderResult",
    "OrderStatus",
    "OrderType",
    "SmartExecutor",
    "SmartOrderRouter",
    "OrderFillVerifier",
    "LatencyOptimizer",
    "ExecutionStats",
]