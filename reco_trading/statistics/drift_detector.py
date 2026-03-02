from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class CUSUMDrift:
    def __init__(self, k: float = 0.001, h: float = 0.05, mean_alpha: float = 0.05) -> None:
        self.k = float(max(k, 0.0))
        self.h = float(max(h, 1e-9))
        self.mean_alpha = float(np.clip(mean_alpha, 0.001, 1.0))
        self.running_mean = 0.0
        self.cusum = 0.0
        self.initialized = False

    def update(self, value: float) -> float:
        x = float(value)
        if not self.initialized:
            self.running_mean = x
            self.initialized = True
            self.cusum = 0.0
            return 0.0

        self.running_mean = ((1.0 - self.mean_alpha) * self.running_mean) + (self.mean_alpha * x)
        self.cusum = float(max(0.0, self.cusum + (x - self.running_mean - self.k)))
        return float(np.clip(self.cusum / self.h, 0.0, 1.0))


class KLDivergenceDrift:
    def __init__(self, bins: int = 20, epsilon: float = 1e-9) -> None:
        self.bins = int(max(bins, 5))
        self.epsilon = float(max(epsilon, 1e-12))

    def compute(self, recent: np.ndarray, historical: np.ndarray) -> float:
        recent_arr = np.asarray(recent, dtype=float)
        historical_arr = np.asarray(historical, dtype=float)
        recent_arr = recent_arr[np.isfinite(recent_arr)]
        historical_arr = historical_arr[np.isfinite(historical_arr)]

        if recent_arr.size < 5 or historical_arr.size < 5:
            return 0.0

        min_edge = float(min(recent_arr.min(), historical_arr.min()))
        max_edge = float(max(recent_arr.max(), historical_arr.max()))
        if not np.isfinite(min_edge) or not np.isfinite(max_edge) or max_edge <= min_edge:
            return 0.0

        bins = np.linspace(min_edge, max_edge, self.bins + 1)
        p_hist, _ = np.histogram(recent_arr, bins=bins, density=True)
        q_hist, _ = np.histogram(historical_arr, bins=bins, density=True)
        p = p_hist + self.epsilon
        q = q_hist + self.epsilon
        p /= p.sum()
        q /= q.sum()

        return float(np.sum(p * np.log(p / q)))


@dataclass(slots=True)
class DriftSignal:
    drift_score: float
    regime_change_probability: float
