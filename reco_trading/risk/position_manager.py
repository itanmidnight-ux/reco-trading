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


class PositionManager:
    """Tracks active position and exit conditions."""

    def __init__(self) -> None:
        self.positions: list[Position] = []

    def can_open(self, confidence: float) -> bool:
        max_positions = 2 if confidence >= 0.90 else 1
        return len(self.positions) < max_positions

    def open(self, position: Position) -> None:
        self.positions.append(position)

    def check_exit(self, position: Position, current_price: float) -> str | None:
        if current_price <= position.stop_loss:
            return "STOP_LOSS_HIT"
        if current_price >= position.take_profit:
            return "TAKE_PROFIT_HIT"
        if current_price >= position.entry_price + position.atr:
            trail = current_price - position.atr
            position.trailing_stop = max(position.trailing_stop or 0.0, trail)
        if position.trailing_stop and current_price <= position.trailing_stop:
            return "TRAILING_STOP_HIT"
        return None

    def close(self, trade_id: int) -> Position | None:
        for idx, position in enumerate(self.positions):
            if position.trade_id == trade_id:
                return self.positions.pop(idx)
        return None
