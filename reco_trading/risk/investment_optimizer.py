from __future__ import annotations

from dataclasses import dataclass
from math import pow

from reco_trading.risk.capital_profile import CapitalProfile


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), low), high)


@dataclass(slots=True)
class OptimizedInvestmentControls:
    risk_per_trade_fraction: float
    max_trade_balance_fraction: float
    capital_reserve_ratio: float
    min_cash_buffer_usdt: float
    capital_limit_usdt: float
    dynamic_exit_enabled: bool
    confidence_boost_multiplier: float
    optimization_reason: str


class InvestmentOptimizer:
    """Produces equity-aware and regime-aware runtime investment controls."""

    def optimize(
        self,
        *,
        equity: float,
        profile: CapitalProfile,
        volatility_ratio: float,
        drawdown_fraction: float,
        win_rate: float | None,
        risk_cap: float | None = None,
        allocation_cap: float | None = None,
    ) -> OptimizedInvestmentControls:
        safe_equity = max(float(equity), 0.0)
        safe_vol = max(float(volatility_ratio), 0.0)
        safe_drawdown = _clamp(drawdown_fraction, 0.0, 0.80)
        equity_denominator = max(safe_equity + 1_000.0, 1.0)
        equity_risk_scale = _clamp(pow(1_000.0 / equity_denominator, 0.16), 0.45, 1.20)
        equity_alloc_scale = _clamp(pow(2_000.0 / (safe_equity + 2_000.0), 0.22), 0.35, 1.15)

        vol_penalty = _clamp(1.02 - (safe_vol * 18.0), 0.55, 1.02)
        drawdown_penalty = _clamp(1.0 - (safe_drawdown * 2.2), 0.40, 1.0)
        performance_factor = 1.0
        if win_rate is not None:
            safe_wr = _clamp(win_rate, 0.0, 1.0)
            if safe_wr >= 0.58:
                performance_factor = 1.08
            elif safe_wr < 0.45:
                performance_factor = 0.90

        risk_cap_value = profile.risk_per_trade_fraction if risk_cap is None else min(float(risk_cap), profile.risk_per_trade_fraction)
        allocation_cap_value = (
            profile.max_trade_balance_fraction
            if allocation_cap is None
            else min(float(allocation_cap), profile.max_trade_balance_fraction)
        )

        raw_risk = profile.risk_per_trade_fraction * 0.95 * vol_penalty * drawdown_penalty * performance_factor * equity_risk_scale
        risk_fraction = _clamp(raw_risk, 0.001, max(risk_cap_value, 0.001))

        raw_allocation = profile.max_trade_balance_fraction * 0.92 * vol_penalty * drawdown_penalty * equity_alloc_scale
        allocation_fraction = _clamp(raw_allocation, 0.01, max(allocation_cap_value, 0.01))

        if safe_equity < 25.0:
            risk_fraction = min(risk_fraction, 0.0035)
            allocation_fraction = min(allocation_fraction, 0.12)
        elif safe_equity > 250_000.0:
            risk_fraction = min(risk_fraction, 0.0030)
            allocation_fraction = min(allocation_fraction, 0.07)
        elif safe_equity > 50_000.0:
            risk_fraction = min(risk_fraction, 0.0045)
            allocation_fraction = min(allocation_fraction, 0.10)

        reserve_ratio = _clamp(profile.reserve_ratio + safe_drawdown * 0.30 + safe_vol * 4.0, profile.reserve_ratio, 0.90)
        cash_buffer = max(profile.reserve_buffer_usdt, safe_equity * (0.015 + safe_vol * 1.5))
        operable_limit = max(safe_equity - max((safe_equity * reserve_ratio), cash_buffer), 0.0)
        if safe_equity > 0 and operable_limit <= 0:
            operable_limit = safe_equity * 0.05

        confidence_boost = 1.0
        if win_rate is not None:
            confidence_boost = _clamp(1.0 + ((float(win_rate) - 0.50) * 0.50), 0.90, 1.15)

        reason = (
            f"vol_penalty={vol_penalty:.2f} drawdown_penalty={drawdown_penalty:.2f} "
            f"equity_risk_scale={equity_risk_scale:.2f} equity_alloc_scale={equity_alloc_scale:.2f} "
            f"performance_factor={performance_factor:.2f}"
        )

        return OptimizedInvestmentControls(
            risk_per_trade_fraction=risk_fraction,
            max_trade_balance_fraction=allocation_fraction,
            capital_reserve_ratio=reserve_ratio,
            min_cash_buffer_usdt=cash_buffer,
            capital_limit_usdt=operable_limit,
            dynamic_exit_enabled=True,
            confidence_boost_multiplier=confidence_boost,
            optimization_reason=reason,
        )
