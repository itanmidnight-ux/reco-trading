from __future__ import annotations

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
    """Tracks active position and exit conditions with partial take profit and safety orders."""

    def __init__(self) -> None:
        self.positions: list[Position] = []

    def can_open(self, max_concurrent_trades: int) -> bool:
        return len(self.positions) < max(int(max_concurrent_trades), 1)

    def open(self, position: Position) -> None:
        if position.initial_risk_distance <= 0:
            position.initial_risk_distance = abs(position.entry_price - position.stop_loss)
        position.peak_price = position.entry_price
        position.partial_tp_triggered = []
        position.safety_orders_triggered = 0
        position.entry_timestamp_ms = position.last_candle_ts_ms or int(time.time() * 1000)
        self.positions.append(position)

    def check_exit(self, position: Position, current_price: float, *, equity: float = 0.0) -> str | None:
        atr = max(position.atr, position.entry_price * 0.002)
        risk_distance = max(position.initial_risk_distance, 1e-9)
        
        position.bars_held += 1

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

    def should_trigger_safety_order(self, position: Position, current_price: float) -> bool:
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

    def close(self, trade_id: int) -> Position | None:
        for idx, position in enumerate(self.positions):
            if position.trade_id == trade_id:
                return self.positions.pop(idx)
        return None
