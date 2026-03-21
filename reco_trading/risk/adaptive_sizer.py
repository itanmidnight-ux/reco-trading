from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SizingDecision:
    quantity: float
    risk_amount: float
    size_multiplier: float
    reason: str


class AdaptiveSizer:
    """Adaptive position sizing driven by confidence, volatility and recent results."""

    def __init__(
        self,
        base_risk_fraction: float = 0.01,
        min_multiplier: float = 0.30,
        max_multiplier: float = 1.50,
        confidence_boost_above: float = 0.80,
    ) -> None:
        self.base_risk_fraction = max(float(base_risk_fraction), 0.001)
        self.min_multiplier = max(float(min_multiplier), 0.10)
        self.max_multiplier = min(float(max_multiplier), 2.0)
        self.confidence_boost_above = float(confidence_boost_above)

    def compute(
        self,
        *,
        equity: float,
        price: float,
        stop_loss: float,
        atr: float,
        confidence: float,
        recent_pnls: list[float],
        volatility_multiplier: float = 1.0,
    ) -> SizingDecision:
        safe_equity = max(float(equity), 0.0)
        safe_price = max(float(price), 1e-9)
        risk_amount = safe_equity * self.base_risk_fraction

        raw_distance = abs(safe_price - float(stop_loss))
        atr_floor = max(float(atr) * 0.5, safe_price * 0.0005)
        stop_distance = max(raw_distance, atr_floor, 1e-9)

        base_qty = (risk_amount / stop_distance) / safe_price

        conf = max(0.0, min(1.0, float(confidence)))
        if conf >= self.confidence_boost_above:
            conf_mult = 1.0 + 0.4 * ((conf - self.confidence_boost_above) / (1.0 - self.confidence_boost_above))
        elif conf >= 0.60:
            conf_mult = 0.85 + 0.15 * ((conf - 0.60) / (self.confidence_boost_above - 0.60))
        else:
            conf_mult = 0.60

        streak_mult = 1.0
        if len(recent_pnls) >= 2:
            last = recent_pnls[-3:] if len(recent_pnls) >= 3 else recent_pnls
            wins = sum(1 for pnl in last if pnl > 0)
            losses = sum(1 for pnl in last if pnl < 0)
            if wins == len(last):
                streak_mult = 1.10
            elif losses >= 2:
                streak_mult = 0.75

        final_mult = conf_mult * streak_mult * max(float(volatility_multiplier), 0.30)
        final_mult = max(self.min_multiplier, min(self.max_multiplier, final_mult))
        final_qty = max(base_qty * final_mult, 0.0)

        reason = (
            f"conf={conf:.2f} conf_mult={conf_mult:.2f} "
            f"streak_mult={streak_mult:.2f} vol_mult={volatility_multiplier:.2f} "
            f"final_mult={final_mult:.2f}"
        )

        return SizingDecision(
            quantity=final_qty,
            risk_amount=risk_amount,
            size_multiplier=final_mult,
            reason=reason,
        )
