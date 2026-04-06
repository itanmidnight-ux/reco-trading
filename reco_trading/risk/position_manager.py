from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Position:
    trade_id: int
    side: str
    quantity: float
    entry_price: float
    stop_loss: float
    take_profit: float
    atr: float
    trailing_stop: float | None = None
    initial_risk_distance: float = 0.0
    dynamic_exit_enabled: bool = False
    peak_price: float | None = None
    peak_retrace_atr: float = 0.85
    structure_exit_votes: int = 0
    bars_held: int = 0
    last_candle_ts_ms: int | None = None
    partial_tp_triggered: list[dict] = field(default_factory=list)
    safety_orders_triggered: int = 0
    entry_timestamp_ms: int | None = None


class PositionManager:
    """Tracks active position and exit conditions with partial take profit and safety orders.

    Thread-safe implementation using asyncio.Lock for all position operations.
    """

    def __init__(self) -> None:
        self._positions: list[Position] = []
        self._lock = asyncio.Lock()

    @property
    def positions(self) -> list[Position]:
        """Return a copy of positions list for safe iteration."""
        return list(self._positions)

    async def can_open(self, max_concurrent_trades: int) -> bool:
        async with self._lock:
            return len(self._positions) < max(int(max_concurrent_trades), 1)

    async def open(self, position: Position) -> None:
        async with self._lock:
            if position.initial_risk_distance <= 0:
                position.initial_risk_distance = abs(position.entry_price - position.stop_loss)
            position.peak_price = position.entry_price
            position.partial_tp_triggered = []
            position.safety_orders_triggered = 0
            position.entry_timestamp_ms = position.last_candle_ts_ms or int(time.time() * 1000)
            self._positions.append(position)

    async def check_exit(self, position: Position, current_price: float, *, equity: float = 0.0) -> str | None:
        async with self._lock:
            atr = max(position.atr, position.entry_price * 0.002)
            risk_distance = max(position.initial_risk_distance, 1e-9)
            
            # Find the position in our list and update it
            pos_ref = None
            for p in self._positions:
                if p.trade_id == position.trade_id:
                    pos_ref = p
                    break
            
            if pos_ref:
                pos_ref.bars_held += 1
                position.bars_held = pos_ref.bars_held
            
            if not position.dynamic_exit_enabled:
                partial_tp_exit = self._check_partial_take_profit(position, current_price, atr)
                if partial_tp_exit:
                    return partial_tp_exit

            if position.side == "BUY":
                return self._check_exit_buy(position, current_price, atr, risk_distance)
            else:
                return self._check_exit_sell(position, current_price, atr, risk_distance)

    def _check_partial_take_profit(self, position: Position, current_price: float, atr: float) -> str | None:
        if position.side == "BUY":
            profit_pct = (current_price - position.entry_price) / position.entry_price
        else:
            profit_pct = (position.entry_price - current_price) / position.entry_price

        if profit_pct <= 0:
            return None

        tp_levels = [
            {"level": 1, "profit_pct": 0.015, "quantity_pct": 0.30, "id": "TP1_1.5%"},
            {"level": 2, "profit_pct": 0.030, "quantity_pct": 0.30, "id": "TP2_3.0%"},
            {"level": 3, "profit_pct": 0.050, "quantity_pct": 0.25, "id": "TP3_5.0%"},
        ]

        for tp in tp_levels:
            if tp["id"] in position.partial_tp_triggered:
                continue
            
            if profit_pct >= tp["profit_pct"]:
                position.partial_tp_triggered.append(tp["id"])
                return f"PARTIAL_TP_{tp['id']}"

        return None

    def _check_exit_buy(self, position: Position, current_price: float, atr: float, risk_distance: float) -> str | None:
        position.peak_price = max(position.peak_price or position.entry_price, current_price)
        if current_price <= position.stop_loss:
            return "STOP_LOSS_HIT"
        if not position.dynamic_exit_enabled and current_price >= position.take_profit:
            return "TAKE_PROFIT_HIT"
        
        if position.dynamic_exit_enabled:
            retrace_distance = max(position.peak_retrace_atr * atr, risk_distance * 0.60)
            if (position.peak_price - current_price) >= retrace_distance and current_price > position.entry_price:
                return "PEAK_RETRACE_EXIT"

        profit = current_price - position.entry_price
        profit_in_atr = profit / max(atr, 1e-9)
        
        trailing_distance = 0.8 * atr
        if profit_in_atr > 3.5:
            trailing_distance = 0.4 * atr
        elif profit_in_atr > 2.5:
            trailing_distance = 0.55 * atr
        elif profit_in_atr > 1.5:
            trailing_distance = 0.65 * atr
            
        activation_distance = max(0.8 * atr, risk_distance)
        if profit >= activation_distance:
            trail = current_price - trailing_distance
            position.trailing_stop = max(position.trailing_stop or position.stop_loss, trail)
            
        if position.trailing_stop and current_price <= position.trailing_stop:
            return "TRAILING_STOP_HIT"
        return None

    def _check_exit_sell(self, position: Position, current_price: float, atr: float, risk_distance: float) -> str | None:
        position.peak_price = min(position.peak_price or position.entry_price, current_price)
        if current_price >= position.stop_loss:
            return "STOP_LOSS_HIT"
        if not position.dynamic_exit_enabled and current_price <= position.take_profit:
            return "TAKE_PROFIT_HIT"
        
        if position.dynamic_exit_enabled:
            retrace_distance = max(position.peak_retrace_atr * atr, risk_distance * 0.60)
            if (current_price - position.peak_price) >= retrace_distance and current_price < position.entry_price:
                return "PEAK_RETRACE_EXIT"

        profit = position.entry_price - current_price
        profit_in_atr = profit / max(atr, 1e-9)
        
        trailing_distance = 0.8 * atr
        if profit_in_atr > 3.5:
            trailing_distance = 0.4 * atr
        elif profit_in_atr > 2.5:
            trailing_distance = 0.55 * atr
        elif profit_in_atr > 1.5:
            trailing_distance = 0.65 * atr
            
        activation_distance = max(0.8 * atr, risk_distance)
        if profit >= activation_distance:
            trail = current_price + trailing_distance
            position.trailing_stop = min(position.trailing_stop or position.stop_loss, trail)
            
        if position.trailing_stop and current_price >= position.trailing_stop:
            return "TRAILING_STOP_HIT"
        return None

    async def should_trigger_safety_order(self, position: Position, current_price: float) -> bool:
        async with self._lock:
            if position.side == "BUY":
                drawdown = (position.entry_price - current_price) / position.entry_price
            else:
                drawdown = (current_price - position.entry_price) / position.entry_price

            if position.safety_orders_triggered >= 2:
                return False

            if drawdown >= 0.015 and position.safety_orders_triggered == 0:
                return True
            if drawdown >= 0.030 and position.safety_orders_triggered == 1:
                return True
                
            return False

    async def close(self, trade_id: int) -> Position | None:
        async with self._lock:
            for idx, position in enumerate(self._positions):
                if position.trade_id == trade_id:
                    return self._positions.pop(idx)
            return None

    async def get_position(self, trade_id: int) -> Position | None:
        async with self._lock:
            for position in self._positions:
                if position.trade_id == trade_id:
                    return position
            return None

    async def get_all_positions(self) -> list[Position]:
        async with self._lock:
            return list(self._positions)

    async def update_position(self, trade_id: int, **kwargs: Any) -> bool:
        async with self._lock:
            for position in self._positions:
                if position.trade_id == trade_id:
                    for key, value in kwargs.items():
                        if hasattr(position, key):
                            setattr(position, key, value)
                    return True
            return False

    def count(self) -> int:
        """Return count without lock for fast access (read-only)."""
        return len(self._positions)