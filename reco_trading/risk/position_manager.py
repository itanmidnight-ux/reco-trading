from __future__ import annotations

from dataclasses import dataclass


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


class PositionManager:
    """Tracks active position and exit conditions."""

    def __init__(self) -> None:
        self.positions: list[Position] = []

    def can_open(self, max_concurrent_trades: int) -> bool:
        return len(self.positions) < max(int(max_concurrent_trades), 1)

    def open(self, position: Position) -> None:
        if position.initial_risk_distance <= 0:
            position.initial_risk_distance = abs(position.entry_price - position.stop_loss)
        self.positions.append(position)

    def check_exit(self, position: Position, current_price: float) -> str | None:
        atr = max(position.atr, position.entry_price * 0.002)
        risk_distance = max(position.initial_risk_distance, 1e-9)
        activation_distance = 1.5 * atr
        trailing_distance = 1.2 * atr

        if position.side == "BUY":
            if current_price <= position.stop_loss:
                return "STOP_LOSS_HIT"
            if current_price >= position.take_profit:
                return "TAKE_PROFIT_HIT"

            profit = current_price - position.entry_price
            if profit >= max(activation_distance, risk_distance):
                trail = current_price - trailing_distance
                position.trailing_stop = max(position.trailing_stop or position.stop_loss, trail)
            if position.trailing_stop and current_price <= position.trailing_stop:
                return "TRAILING_STOP_HIT"
            return None

        if current_price >= position.stop_loss:
            return "STOP_LOSS_HIT"
        if current_price <= position.take_profit:
            return "TAKE_PROFIT_HIT"

        profit = position.entry_price - current_price
        if profit >= max(activation_distance, risk_distance):
            trail = current_price + trailing_distance
            position.trailing_stop = min(position.trailing_stop or position.stop_loss, trail)
        if position.trailing_stop and current_price >= position.trailing_stop:
            return "TRAILING_STOP_HIT"
        return None

    def close(self, trade_id: int) -> Position | None:
        for idx, position in enumerate(self.positions):
            if position.trade_id == trade_id:
                return self.positions.pop(idx)
        return None
