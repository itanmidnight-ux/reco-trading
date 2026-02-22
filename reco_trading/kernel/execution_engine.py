from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Friction:
    fees: float
    slippage: float
    spread: float

    @property
    def total_cost(self) -> float:
        return self.fees + self.slippage + self.spread


class ExecutionGate:
    def __init__(self, min_timeframe_seconds: int = 60) -> None:
        self.min_timeframe_seconds = min_timeframe_seconds

    def allow_trade(self, *, expected_edge: float, friction: Friction, timeframe_seconds: int, atr: float, price: float) -> tuple[bool, str]:
        if timeframe_seconds < self.min_timeframe_seconds:
            return False, 'timeframe_too_low_for_spot'
        if atr <= max(price * 0.0005, 1e-9):
            return False, 'atr_too_low'
        if expected_edge <= friction.total_cost:
            return False, 'edge_below_total_friction'
        if friction.spread >= abs(expected_edge):
            return False, 'spread_dominates_edge'
        return True, 'ok'
