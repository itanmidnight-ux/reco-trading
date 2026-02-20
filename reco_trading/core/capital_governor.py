from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CapitalGovernorState:
    base_capital: float = 1.0
    capital_multiplier: float = 1.0


class CapitalGovernor:
    """Gobierna el capital disponible para ejecución en función del riesgo live."""

    def __init__(self, base_capital: float = 1.0) -> None:
        self.state = CapitalGovernorState(base_capital=max(base_capital, 0.0), capital_multiplier=1.0)

    @property
    def deployable_capital(self) -> float:
        return self.state.base_capital * self.state.capital_multiplier

    def reduce_capital(self, target_multiplier: float) -> float:
        clipped = float(min(1.0, max(0.05, target_multiplier)))
        self.state.capital_multiplier = min(self.state.capital_multiplier, clipped)
        return self.state.capital_multiplier

    def reset(self) -> None:
        self.state.capital_multiplier = 1.0
