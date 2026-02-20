from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(slots=True)
class RiskLimits:
    risk_per_trade: float
    max_daily_loss: float
    max_global_drawdown: float
    max_total_exposure: float
    max_asset_exposure: float
    correlation_threshold: float


@dataclass(slots=True)
class RiskAssessment:
    allowed: bool
    reason: str
    position_size: float
    stop_price: float
    kelly_fraction: float
    reward_risk_ratio: float
    reduced_by_correlation: bool = False


@dataclass(slots=True)
class RiskState:
    peak_equity: float = 0.0
    negative_streak: int = 0
    kill_switch: bool = False
    asset_exposure: dict[str, float] = field(default_factory=dict)


class InstitutionalRiskManager:
    def __init__(self, limits: RiskLimits, atr_multiplier: float = 2.0, max_streak: int = 6) -> None:
        self.limits = limits
        self.atr_multiplier = atr_multiplier
        self.max_streak = max_streak
        self.state = RiskState(peak_equity=1.0)

    @staticmethod
    def _modified_kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
        # f* = p - (1-p)/b, con b = AvgWin/AvgLoss
        b = max(avg_win / max(avg_loss, 1e-9), 1e-6)
        raw = win_rate - (1.0 - win_rate) / b
        return float(np.clip(raw * 0.5, 0.0, 0.25))

    def _drawdown(self, equity: float) -> float:
        self.state.peak_equity = max(self.state.peak_equity, equity)
        return 1.0 - (equity / max(self.state.peak_equity, 1e-9))

    def _correlation_penalty(self, symbol: str, returns: pd.DataFrame) -> tuple[float, bool]:
        if symbol not in returns.columns or len(returns.columns) < 2:
            return 1.0, False
        corr = returns.corr().fillna(0.0)
        c = corr[symbol].drop(symbol)
        if c.empty:
            return 1.0, False
        max_corr = float(c.abs().max())
        if max_corr <= self.limits.correlation_threshold:
            return 1.0, False
        penalty = float(np.clip(1.0 - (max_corr - self.limits.correlation_threshold), 0.3, 1.0))
        return penalty, True

    def assess(
        self,
        *,
        symbol: str,
        side: str,
        equity: float,
        daily_pnl: float,
        current_price: float,
        atr: float,
        annualized_volatility: float,
        volatility_multiplier: float,
        expected_win_rate: float,
        avg_win: float,
        avg_loss: float,
        returns_matrix: pd.DataFrame,
    ) -> RiskAssessment:
        if self.state.kill_switch:
            return RiskAssessment(False, "kill_switch_enabled", 0.0, current_price, 0.0, 0.0)

        if daily_pnl <= -abs(equity * self.limits.max_daily_loss):
            self.state.kill_switch = True
            return RiskAssessment(False, "max_daily_loss_hit", 0.0, current_price, 0.0, 0.0)

        drawdown = self._drawdown(equity)
        if drawdown >= self.limits.max_global_drawdown:
            self.state.kill_switch = True
            return RiskAssessment(False, "max_global_drawdown_hit", 0.0, current_price, 0.0, 0.0)

        if atr <= 0:
            return RiskAssessment(False, "invalid_atr", 0.0, current_price, 0.0, 0.0)

        kelly_fraction = self._modified_kelly(expected_win_rate, avg_win, avg_loss)
        risk_budget = equity * self.limits.risk_per_trade * max(kelly_fraction, 0.05)

        # PositionSize = (RiskPerTrade * Equity) / (ATR * Multiplier)
        base_size = risk_budget / max(atr * self.atr_multiplier, 1e-9)
        vol_adj = float(np.clip(1.0 / max(annualized_volatility, 0.10), 0.2, 1.2))
        size = base_size * vol_adj * float(np.clip(volatility_multiplier, 0.0, 1.0))

        corr_penalty, corr_reduced = self._correlation_penalty(symbol, returns_matrix)
        size *= corr_penalty

        max_asset = equity * self.limits.max_asset_exposure / max(current_price, 1e-9)
        size = min(size, max_asset)

        stop_distance = atr * self.atr_multiplier
        stop_price = current_price - stop_distance if side == "BUY" else current_price + stop_distance

        reward_risk = float(np.clip(1.5 + (1.0 - annualized_volatility), 1.0, 3.0))

        if size <= 0:
            return RiskAssessment(False, "size_capped_to_zero", 0.0, stop_price, kelly_fraction, reward_risk, corr_reduced)

        return RiskAssessment(True, "ok", float(size), stop_price, kelly_fraction, reward_risk, corr_reduced)

    def register_trade_result(self, pnl: float) -> None:
        self.state.negative_streak = self.state.negative_streak + 1 if pnl < 0 else 0
        if self.state.negative_streak >= self.max_streak:
            self.state.kill_switch = True
