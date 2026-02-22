from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class EdgeModelInput:
    momentum_z: float
    volatility_regime: float
    volume_score: float
    spread_cost: float
    rsi_bias: float


class ProbabilityModel:
    """Logistic model with exponential smoothing to avoid confidence jumps."""

    def __init__(self, smoothing_alpha: float = 0.2) -> None:
        self._alpha = float(np.clip(smoothing_alpha, 0.01, 1.0))
        self._last_probability = 0.5

    @staticmethod
    def _sigmoid(value: float) -> float:
        clipped = float(np.clip(value, -30.0, 30.0))
        return float(1.0 / (1.0 + np.exp(-clipped)))

    def infer_edge_probability(self, data: EdgeModelInput) -> float:
        linear = (
            1.25 * data.momentum_z
            - 0.9 * data.volatility_regime
            + 0.8 * data.volume_score
            - 1.1 * data.spread_cost
            + 0.5 * data.rsi_bias
        )
        raw_probability = self._sigmoid(linear)
        smoothed = (self._alpha * raw_probability) + ((1.0 - self._alpha) * self._last_probability)
        self._last_probability = float(np.clip(smoothed, 0.0, 1.0))
        return self._last_probability
