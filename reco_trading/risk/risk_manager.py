from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RiskDecision:
    approved: bool
    reason: str


@dataclass(slots=True)
class PositionSizing:
    quantity: float
    stop_distance: float
    risk_amount: float


class RiskManager:
    """Capital protection and trading guardrails."""

    def __init__(self, max_daily_loss_fraction: float, max_trades_per_day: int) -> None:
        self.max_daily_loss_fraction = max_daily_loss_fraction
        self.max_trades_per_day = max_trades_per_day

    def validate(
        self,
        balance: float,
        daily_pnl: float,
        trades_today: int,
        confidence: float,
        confidence_threshold: float,
    ) -> RiskDecision:
        if trades_today >= self.max_trades_per_day:
            return RiskDecision(False, "MAX_TRADES_PER_DAY")
        if daily_pnl <= -(balance * self.max_daily_loss_fraction):
            return RiskDecision(False, "RISK_PAUSE")
        if confidence < confidence_threshold:
            return RiskDecision(False, "LOW_CONFIDENCE")
        return RiskDecision(True, "OK")

    def position_size_for_risk(self, equity: float, risk_fraction: float, price: float, atr: float) -> PositionSizing:
        risk_amount = max(equity * risk_fraction, 0.0)
        stop_distance = max(float(atr) * 1.5, float(price) * 0.002)
        if risk_amount <= 0 or stop_distance <= 0:
            return PositionSizing(0.0, stop_distance, risk_amount)
        quantity = risk_amount / stop_distance
        return PositionSizing(quantity=quantity, stop_distance=stop_distance, risk_amount=risk_amount)
