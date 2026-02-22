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


@dataclass(frozen=True, slots=True)
class PostOrderCheck:
    ok: bool
    reason: str
    expected_price: float
    executed_price: float
    expected_slippage: float
    realized_slippage: float
    fill_ratio: float
    status: str


class ExecutionGate:
    def __init__(self, min_timeframe_seconds: int = 60, max_slippage_multiplier: float = 2.0) -> None:
        self.min_timeframe_seconds = min_timeframe_seconds
        self.max_slippage_multiplier = max(float(max_slippage_multiplier), 1.0)

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

    def validate_post_order(
        self,
        *,
        status: str,
        requested_qty: float,
        executed_qty: float,
        expected_price: float,
        executed_price: float,
        expected_slippage: float,
    ) -> PostOrderCheck:
        normalized_status = str(status or 'unknown').lower()
        safe_requested = max(float(requested_qty), 1e-9)
        fill_ratio = float(max(float(executed_qty), 0.0) / safe_requested)
        realized_slippage = float(abs(float(executed_price) - float(expected_price)) / max(float(expected_price), 1e-9))
        safe_expected_slippage = max(float(expected_slippage), 1e-9)

        if normalized_status in {'rejected', 'expired', 'canceled', 'cancelled'}:
            return PostOrderCheck(
                ok=False,
                reason=f'order_{normalized_status}',
                expected_price=float(expected_price),
                executed_price=float(executed_price),
                expected_slippage=float(expected_slippage),
                realized_slippage=realized_slippage,
                fill_ratio=fill_ratio,
                status=normalized_status,
            )

        if fill_ratio < 1.0:
            return PostOrderCheck(
                ok=False,
                reason='partial_fill',
                expected_price=float(expected_price),
                executed_price=float(executed_price),
                expected_slippage=float(expected_slippage),
                realized_slippage=realized_slippage,
                fill_ratio=fill_ratio,
                status=normalized_status,
            )

        if realized_slippage > (safe_expected_slippage * self.max_slippage_multiplier):
            return PostOrderCheck(
                ok=False,
                reason='slippage_exceeded',
                expected_price=float(expected_price),
                executed_price=float(executed_price),
                expected_slippage=float(expected_slippage),
                realized_slippage=realized_slippage,
                fill_ratio=fill_ratio,
                status=normalized_status,
            )

        return PostOrderCheck(
            ok=True,
            reason='ok',
            expected_price=float(expected_price),
            executed_price=float(executed_price),
            expected_slippage=float(expected_slippage),
            realized_slippage=realized_slippage,
            fill_ratio=fill_ratio,
            status=normalized_status,
        )
