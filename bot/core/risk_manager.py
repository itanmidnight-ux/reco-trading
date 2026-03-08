from __future__ import annotations

from dataclasses import dataclass

from bot.config import BotConfig
from bot.core.portfolio import Portfolio


@dataclass
class RiskDecision:
    approved: bool
    reason: str = ''


class RiskManager:
    def __init__(self, config: BotConfig) -> None:
        self.config = config

    def validate_new_trade(self, portfolio: Portfolio) -> RiskDecision:
        if portfolio.open_position is not None:
            return RiskDecision(False, 'max_open_positions reached')
        if portfolio.daily_trades >= self.config.max_trades_per_day:
            return RiskDecision(False, 'max_trades_per_day reached')
        if portfolio.daily_loss >= self.config.daily_loss_limit:
            return RiskDecision(False, 'daily_loss_limit reached, trading paused')
        return RiskDecision(True)

    def position_size_from_risk(self, equity: float, entry_price: float, stop_loss_price: float) -> float:
        max_loss = equity * self.config.max_risk_per_trade
        risk_per_unit = abs(entry_price - stop_loss_price)
        if risk_per_unit <= 0:
            return 0.0
        return max_loss / risk_per_unit


__all__ = ['RiskManager', 'RiskDecision']
