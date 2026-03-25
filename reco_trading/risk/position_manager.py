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
    dynamic_exit_enabled: bool = False
    peak_price: float | None = None
    peak_retrace_atr: float = 0.85
    structure_exit_votes: int = 0
    bars_held: int = 0
    last_candle_ts_ms: int | None = None


class PositionManager:
    """Tracks active position and exit conditions."""

    def __init__(self) -> None:
        self.positions: list[Position] = []

    def can_open(self, max_concurrent_trades: int) -> bool:
        return len(self.positions) < max(int(max_concurrent_trades), 1)

    def open(self, position: Position) -> None:
        if position.initial_risk_distance <= 0:
            position.initial_risk_distance = abs(position.entry_price - position.stop_loss)
        position.peak_price = position.entry_price
        self.positions.append(position)

    def check_exit(self, position: Position, current_price: float) -> str | None:
        atr = max(position.atr, position.entry_price * 0.002)
        risk_distance = max(position.initial_risk_distance, 1e-9)
        activation_distance = 1.0 * atr
        trailing_distance = 0.8 * atr

        if position.side == "BUY":
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
            if profit_in_atr > 3.0:
                trailing_distance = 0.5 * atr
            elif profit_in_atr > 2.0:
                trailing_distance = 0.65 * atr
            if profit >= max(activation_distance, risk_distance):
                trail = current_price - trailing_distance
                position.trailing_stop = max(position.trailing_stop or position.stop_loss, trail)
            if position.trailing_stop and current_price <= position.trailing_stop:
                return "TRAILING_STOP_HIT"
            return None

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
