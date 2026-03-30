from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AdvancedRiskDecision:
    approved: bool
    reason: str
    pause_trading: bool
    size_multiplier: float


class AdvancedRiskManager:
    """Advanced account-level risk controls independent from base risk manager."""

    def __init__(
        self,
        max_daily_loss_percent: float = 3.0,
        max_consecutive_losses: int = 3,
        max_drawdown_percent: float = 10.0,
        high_volatility_cut: float = 0.5,
    ) -> None:
        self.max_daily_loss_percent = max_daily_loss_percent
        self.max_consecutive_losses = max_consecutive_losses
        self.max_drawdown_percent = max_drawdown_percent
        self.high_volatility_cut = high_volatility_cut

    def evaluate(
        self,
        *,
        daily_pnl: float,
        starting_equity: float,
        consecutive_losses: int,
        current_equity: float,
        peak_equity: float,
        volatility_ratio: float,
    ) -> AdvancedRiskDecision:
        if starting_equity <= 0:
            starting_equity = 1.0
        if peak_equity <= 0:
            peak_equity = max(current_equity, 1.0)

        daily_loss_pct = abs(min(daily_pnl, 0.0)) / starting_equity * 100
        if daily_loss_pct >= self.max_daily_loss_percent:
            return AdvancedRiskDecision(False, "MAX_DAILY_LOSS", True, 0.0)

        if consecutive_losses >= self.max_consecutive_losses:
            return AdvancedRiskDecision(False, "MAX_CONSECUTIVE_LOSSES", True, 0.0)

        drawdown_pct = max((peak_equity - current_equity) / peak_equity * 100, 0.0)
        if drawdown_pct >= self.max_drawdown_percent:
            return AdvancedRiskDecision(False, "MAX_DRAWDOWN", True, 0.0)

        if volatility_ratio >= 0.02:
            return AdvancedRiskDecision(True, "HIGH_VOLATILITY", False, self.high_volatility_cut)

        if volatility_ratio <= 0.003:
            return AdvancedRiskDecision(True, "LOW_VOLATILITY", False, 0.8)

        return AdvancedRiskDecision(True, "OK", False, 1.0)

    @staticmethod
    def dynamic_position_size(base_size: float, size_multiplier: float) -> float:
        return max(base_size * max(size_multiplier, 0.0), 0.0)
