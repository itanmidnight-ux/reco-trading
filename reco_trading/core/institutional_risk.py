from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class RiskConfig:
    risk_per_trade: float = 0.01
    max_daily_loss: float = 0.05
    max_drawdown: float = 0.20
    max_exposure: float = 0.30
    max_correlation: float = 0.8
    kelly_fraction: float = 0.5
    max_consecutive_losses: int = 5
    confidence_hold_threshold: float = 0.60
    max_confidence_allocation: float = 0.02


class InstitutionalRiskManager:
    def __init__(self, config: RiskConfig):
        self.config = config
        self.equity_peak: float | None = None
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.kill_switch = False

    def update_equity(self, equity: float) -> None:
        if self.equity_peak is None:
            self.equity_peak = equity
        self.equity_peak = max(self.equity_peak, equity)

    def calculate_drawdown(self, equity: float) -> float:
        if self.equity_peak is None:
            self.update_equity(equity)
        return (float(self.equity_peak) - equity) / max(float(self.equity_peak), 1e-9)

    def check_kill_switch(self, equity: float) -> None:
        drawdown = self.calculate_drawdown(equity)

        if drawdown >= self.config.max_drawdown:
            self.kill_switch = True

        if self.daily_pnl <= -self.config.max_daily_loss * equity:
            self.kill_switch = True

        if self.consecutive_losses >= self.config.max_consecutive_losses:
            self.kill_switch = True

    def calculate_position_size(self, equity: float, atr: float, win_rate: float, reward_risk: float) -> float:
        if atr <= 0 or reward_risk <= 0:
            return 0.0

        kelly = (win_rate * reward_risk - (1 - win_rate)) / reward_risk
        kelly = max(kelly, 0)
        kelly *= self.config.kelly_fraction

        risk_capital = equity * self.config.risk_per_trade * kelly
        position_size = risk_capital / (atr + 1e-8)

        return min(position_size, equity * self.config.max_exposure)

    def check_correlation_risk(self, returns_df: pd.DataFrame) -> bool:
        if returns_df.empty:
            return False
        corr_matrix = returns_df.corr()
        high_corr = (corr_matrix > self.config.max_correlation).sum().sum()
        return high_corr > len(corr_matrix)

    def update_trade_result(self, pnl: float) -> None:
        self.daily_pnl += pnl

        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0


    def confidence_position_fraction(self, confidence: float) -> float:
        c = float(confidence)
        if c < self.config.confidence_hold_threshold:
            return 0.0
        if c >= 0.90:
            pct = 0.020
        elif c >= 0.80:
            pct = 0.010
        elif c >= 0.70:
            pct = 0.005
        elif c >= 0.65:
            pct = 0.0025
        else:
            pct = 0.0
        return min(pct, self.config.max_confidence_allocation)
