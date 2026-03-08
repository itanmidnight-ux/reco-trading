from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RiskDecision:
    approved: bool
    reason: str


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
