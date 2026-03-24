from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PortfolioDecision:
    approved: bool
    reason: str


class PortfolioRiskController:
    """Controls portfolio-level exposure for multi-symbol operation."""

    def __init__(
        self,
        *,
        max_global_exposure_fraction: float = 0.7,
        max_symbol_correlation: float = 0.85,
        symbol_caps: dict[str, float] | None = None,
    ) -> None:
        self.max_global_exposure_fraction = max(0.0, min(1.0, float(max_global_exposure_fraction)))
        self.max_symbol_correlation = max(-1.0, min(1.0, float(max_symbol_correlation)))
        self.symbol_caps = {str(k): max(float(v), 0.0) for k, v in (symbol_caps or {}).items() if str(k)}

    def validate(
        self,
        *,
        symbol: str,
        requested_notional: float,
        current_symbol_notional: float,
        total_open_notional: float,
        equity: float,
        max_correlation_observed: float = 0.0,
    ) -> PortfolioDecision:
        if equity <= 0:
            return PortfolioDecision(False, "invalid_equity")

        next_total = max(total_open_notional, 0.0) + max(requested_notional, 0.0)
        global_exposure = next_total / equity
        if global_exposure > self.max_global_exposure_fraction:
            return PortfolioDecision(False, "global_exposure_limit")

        cap = self.symbol_caps.get(symbol)
        if cap is not None and (current_symbol_notional + requested_notional) > cap:
            return PortfolioDecision(False, "symbol_cap_limit")

        if max_correlation_observed > self.max_symbol_correlation:
            return PortfolioDecision(False, "correlation_limit")

        return PortfolioDecision(True, "ok")
