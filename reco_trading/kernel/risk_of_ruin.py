from __future__ import annotations

from dataclasses import dataclass
from math import exp

import numpy as np


@dataclass(slots=True)
class RiskOfRuinSnapshot:
    risk_of_ruin_probability: float
    recommended_risk_multiplier: float


class RiskOfRuinEstimator:
    def __init__(self, floor_multiplier: float = 0.15) -> None:
        self.floor_multiplier = float(np.clip(floor_multiplier, 0.05, 0.5))

    def estimate(
        self,
        *,
        edge: float,
        variance: float,
        position_fraction: float,
        capital: float,
    ) -> RiskOfRuinSnapshot:
        edge_val = float(edge) if np.isfinite(edge) else 0.0
        var_val = float(abs(variance)) if np.isfinite(variance) else 0.0
        frac = float(np.clip(position_fraction, 0.0, 1.0))
        cap = float(max(capital, 1e-9))

        if frac <= 0.0:
            return RiskOfRuinSnapshot(1.0, self.floor_multiplier)

        adj_var = max(var_val * max(frac, 1e-6), 1e-9)
        edge_scale = edge_val / adj_var
        capital_buffer = float(np.clip(cap / max(cap * frac, 1e-9), 1.0, 50.0))

        if edge_scale <= 0.0:
            ruin_prob = float(np.clip(0.65 + min(abs(edge_scale), 1.0) * 0.30, 0.65, 0.99))
        else:
            ruin_prob = float(np.clip(exp(-2.0 * edge_scale * capital_buffer), 0.001, 0.98))

        recommended = float(np.clip((1.0 - ruin_prob) ** 1.25, self.floor_multiplier, 1.0))
        return RiskOfRuinSnapshot(ruin_prob, recommended)
