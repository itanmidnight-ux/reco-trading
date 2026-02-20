from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from reco_trading.core.microstructure import MicrostructureSnapshot
from reco_trading.kernel.capital_governor import CapitalGovernor, CapitalTicket


@dataclass(slots=True)
class RiskLimits:
    risk_per_trade: float
    max_daily_loss: float
    max_global_drawdown: float
    max_total_exposure: float
    max_asset_exposure: float
    correlation_threshold: float
    max_exchange_exposure: float = 1.0
    capital_isolation: dict[str, float] = field(default_factory=dict)
    max_cross_exchange_notional: float = float('inf')


@dataclass(slots=True)
class RiskAssessment:
    allowed: bool
    reason: str
    position_size: float
    stop_price: float
    kelly_fraction: float
    reward_risk_ratio: float
    reduced_by_correlation: bool = False
    cvar_95: float = 0.0


@dataclass(slots=True)
class RiskState:
    peak_equity: float = 0.0
    negative_streak: int = 0
    kill_switch: bool = False
    asset_exposure: dict[str, float] = field(default_factory=dict)
    exchange_exposure: dict[str, float] = field(default_factory=dict)


class InstitutionalRiskManager:
    def __init__(
        self,
        limits: RiskLimits,
        atr_multiplier: float = 2.0,
        max_streak: int = 6,
        capital_governor: CapitalGovernor | None = None,
    ) -> None:
        self.limits = limits
        self.atr_multiplier = atr_multiplier
        self.max_streak = max_streak
        self.capital_governor = capital_governor
        self.state = RiskState(peak_equity=1.0)

    @staticmethod
    def _modified_kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
        b = max(avg_win / max(avg_loss, 1e-9), 1e-6)
        raw = win_rate - (1.0 - win_rate) / b
        return float(np.clip(raw * 0.5, 0.0, 0.25))

    def _drawdown(self, equity: float) -> float:
        self.state.peak_equity = max(self.state.peak_equity, equity)
        return 1.0 - (equity / max(self.state.peak_equity, 1e-9))

    def current_drawdown(self, equity: float) -> float:
        """Retorna drawdown sin modificar estado interno."""
        peak = max(self.state.peak_equity, 1e-9)
        return 1.0 - (equity / peak)

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

    @staticmethod
    def _tail_risk_cvar(returns: pd.Series, alpha: float = 0.95) -> float:
        if returns.empty:
            return 0.0
        q = float(returns.quantile(1.0 - alpha))
        tail = returns[returns <= q]
        if tail.empty:
            return abs(q)
        return float(abs(tail.mean()))

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
        microstructure: MicrostructureSnapshot | None = None,
        risk_per_trade_override: float | None = None,
        exchange: str | None = None,
        notional_by_exchange: dict[str, float] | None = None,
        total_exposure: float | None = None,
        capital_ticket: CapitalTicket | None = None,
    ) -> RiskAssessment:
        if self.capital_governor is not None:
            valid_ticket, ticket_reason = self.capital_governor.validate_ticket(capital_ticket)
            if not valid_ticket:
                return RiskAssessment(False, f'capital_ticket_invalid:{ticket_reason}', 0.0, current_price, 0.0, 0.0)

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

        exchange_notional = abs((notional_by_exchange or {}).get(exchange or '', 0.0))
        if exchange and exchange in self.limits.capital_isolation:
            exchange_cap = abs(equity * self.limits.capital_isolation[exchange])
            if exchange_notional >= exchange_cap:
                return RiskAssessment(False, "capital_isolation_limit_hit", 0.0, current_price, 0.0, 0.0)

        if total_exposure is not None and abs(total_exposure) >= abs(equity * self.limits.max_total_exposure):
            return RiskAssessment(False, "max_total_exposure_hit", 0.0, current_price, 0.0, 0.0)

        if exchange and exchange_notional >= abs(equity * self.limits.max_exchange_exposure):
            return RiskAssessment(False, "max_exchange_exposure_hit", 0.0, current_price, 0.0, 0.0)

        cross_exchange_notional = sum(abs(v) for v in (notional_by_exchange or {}).values())
        if cross_exchange_notional >= abs(self.limits.max_cross_exchange_notional):
            return RiskAssessment(False, "max_cross_exchange_notional_hit", 0.0, current_price, 0.0, 0.0)

        if microstructure and microstructure.liquidity_shock:
            volatility_multiplier *= 0.25

        kelly_fraction = self._modified_kelly(expected_win_rate, avg_win, avg_loss)
        risk_per_trade = float(np.clip(risk_per_trade_override if risk_per_trade_override is not None else self.limits.risk_per_trade, 0.001, 0.10))
        risk_budget = equity * risk_per_trade * max(kelly_fraction, 0.05)

        base_size = risk_budget / max(atr * self.atr_multiplier, 1e-9)
        vol_adj = float(np.clip(1.0 / max(annualized_volatility, 0.10), 0.2, 1.2))
        size = base_size * vol_adj * float(np.clip(volatility_multiplier, 0.0, 1.0))

        if microstructure:
            toxicity_penalty = float(np.clip(1.0 - microstructure.vpin, 0.2, 1.0))
            spread_penalty = float(np.clip(1.0 - 80.0 * microstructure.spread, 0.2, 1.0))
            size *= toxicity_penalty * spread_penalty

        corr_penalty, corr_reduced = self._correlation_penalty(symbol, returns_matrix)
        size *= corr_penalty

        max_asset = equity * self.limits.max_asset_exposure / max(current_price, 1e-9)
        size = min(size, max_asset)

        stop_distance = atr * self.atr_multiplier
        stop_price = current_price - stop_distance if side == "BUY" else current_price + stop_distance

        reward_risk = float(np.clip(1.5 + (1.0 - annualized_volatility), 1.0, 3.0))

        cvar = self._tail_risk_cvar(returns_matrix[symbol]) if symbol in returns_matrix else 0.0
        if cvar > 0.03:
            size *= 0.6

        if exchange and exchange in self.limits.capital_isolation:
            exchange_cap = abs(equity * self.limits.capital_isolation[exchange])
            remaining = max(0.0, exchange_cap - exchange_notional)
            size = min(size, remaining / max(current_price, 1e-9))

        if exchange:
            exchange_cap_units = equity * self.limits.max_exchange_exposure / max(current_price, 1e-9)
            exchange_current_units = exchange_notional / max(current_price, 1e-9)
            size = min(size, max(0.0, exchange_cap_units - exchange_current_units))

        if size <= 0:
            return RiskAssessment(False, "size_capped_to_zero", 0.0, stop_price, kelly_fraction, reward_risk, corr_reduced, cvar)

        return RiskAssessment(True, "ok", float(size), stop_price, kelly_fraction, reward_risk, corr_reduced, cvar)

    def register_trade_result(self, pnl: float) -> None:
        self.state.negative_streak = self.state.negative_streak + 1 if pnl < 0 else 0
        if self.state.negative_streak >= self.max_streak:
            self.state.kill_switch = True
