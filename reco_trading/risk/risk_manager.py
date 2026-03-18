from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RiskDecision:
    approved: bool
    reason: str


@dataclass(slots=True)
class PositionSizing:
    risk_amount: float
    stop_distance: float
    quantity: float


class RiskManager:
    """Capital protection and trading guardrails."""

    def __init__(self, max_daily_loss_fraction: float, max_trades_per_day: int) -> None:
        self.max_daily_loss_fraction = max_daily_loss_fraction
        self.max_trades_per_day = max_trades_per_day

    def validate(
        self,
        balance: float,
        account_equity: float | None,
        daily_pnl: float,
        trades_today: int,
        confidence: float,
        confidence_threshold: float,
    ) -> RiskDecision:
        reference_equity = max(float(account_equity if account_equity is not None else balance), float(balance), 0.0)
        if trades_today >= self.max_trades_per_day:
            return RiskDecision(False, "MAX_TRADES_PER_DAY")
        if daily_pnl <= -(reference_equity * self.max_daily_loss_fraction):
            return RiskDecision(False, "RISK_PAUSE")
        if confidence < confidence_threshold:
            return RiskDecision(False, "LOW_CONFIDENCE")
        return RiskDecision(True, "OK")

    @staticmethod
    def position_size_for_risk(
        *,
        equity: float,
        risk_fraction: float,
        price: float,
        stop_loss_price: float | None = None,
        atr: float = 0.0,
        atr_floor_multiplier: float = 0.5,
    ) -> PositionSizing:
        safe_equity = max(float(equity), 0.0)
        safe_price = max(float(price), 1e-9)
        safe_risk_fraction = max(float(risk_fraction), 0.0)

        risk_amount = safe_equity * safe_risk_fraction
        raw_distance = abs(safe_price - float(stop_loss_price if stop_loss_price is not None else safe_price - (1.5 * atr)))
        atr_floor = max(float(atr) * max(float(atr_floor_multiplier), 0.0), safe_price * 0.0005)
        stop_distance = max(raw_distance, atr_floor, 1e-9)

        position_size = risk_amount / stop_distance if stop_distance > 0 else 0.0
        quantity = position_size / safe_price
        return PositionSizing(risk_amount=risk_amount, stop_distance=stop_distance, quantity=max(quantity, 0.0))
