from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass(slots=True)
class SignalObservation:
    name: str
    score: float
    confidence: float
    regime_weight: float = 1.0
    volatility_adjustment: float = 1.0
    historical_precision: float = 0.5


@dataclass(slots=True)
class FusionResult:
    final_score: float
    probability_up: float
    calibrated_probability: float
    weights: dict[str, float]


@dataclass(slots=True)
class _SignalHistory:
    outcomes: deque[float] = field(default_factory=lambda: deque(maxlen=500))

    def update(self, outcome: float) -> None:
        self.outcomes.append(float(np.clip(outcome, 0.0, 1.0)))

    @property
    def rolling_performance(self) -> float:
        if not self.outcomes:
            return 0.5
        arr = np.asarray(self.outcomes, dtype=float)
        return float(arr.mean())


class SignalFusionEngine:
    """Motor de fusión probabilística institucional.

    Componentes:
    - Normalización robusta por MAD.
    - Rolling performance weighting.
    - Bayesian model averaging (posterior mean Beta-Binomial).
    - Ensemble blending por confidence/regime/volatility.
    - Calibración final con Platt Scaling.
    """

    def __init__(self, model_names: Iterable[str], half_life: int = 64) -> None:
        self._history = {name: _SignalHistory() for name in model_names}
        self._half_life = max(half_life, 2)
        self._platt_a = 1.0
        self._platt_b = 0.0

    @staticmethod
    def _robust_zscore(values: np.ndarray) -> np.ndarray:
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        scale = 1.4826 * mad if mad > 1e-9 else np.std(values) + 1e-9
        return (values - median) / scale

    @staticmethod
    def _sigmoid(x: float) -> float:
        x = float(np.clip(x, -50.0, 50.0))
        return 1.0 / (1.0 + math.exp(-x))

    def update_performance(self, model_name: str, was_correct: bool) -> None:
        if model_name not in self._history:
            self._history[model_name] = _SignalHistory()
        self._history[model_name].update(1.0 if was_correct else 0.0)

    def fit_platt_scaling(self, raw_scores: np.ndarray, labels: np.ndarray, lr: float = 1e-2, epochs: int = 500) -> None:
        """Ajuste de Platt scaling mediante descenso gradiente de log-loss."""
        x = np.asarray(raw_scores, dtype=float)
        y = np.asarray(labels, dtype=float)
        if x.size == 0 or x.size != y.size:
            return

        a, b = self._platt_a, self._platt_b
        for _ in range(epochs):
            logits = np.clip(a * x + b, -40.0, 40.0)
            p = 1.0 / (1.0 + np.exp(-logits))
            grad_a = float(np.mean((p - y) * x))
            grad_b = float(np.mean(p - y))
            a -= lr * grad_a
            b -= lr * grad_b
        self._platt_a = a
        self._platt_b = b

    def _bayesian_weight(self, model_name: str) -> float:
        history = self._history[model_name]
        n = len(history.outcomes)
        if n == 0:
            return 0.5
        k = sum(history.outcomes)
        alpha_post = 1.0 + k
        beta_post = 1.0 + (n - k)
        return alpha_post / (alpha_post + beta_post)

    def fuse(self, observations: list[SignalObservation], meta_weights: dict[str, float] | None = None, meta_confidence: float = 1.0) -> FusionResult:
        if not observations:
            return FusionResult(final_score=0.0, probability_up=0.5, calibrated_probability=0.5, weights={})

        raw_scores = np.asarray([obs.score for obs in observations], dtype=float)
        norm_scores = self._robust_zscore(raw_scores)

        weight_components: list[float] = []
        names: list[str] = []
        for obs in observations:
            rolling_perf = self._history[obs.name].rolling_performance if obs.name in self._history else 0.5
            bayesian_mean = self._bayesian_weight(obs.name) if obs.name in self._history else 0.5
            # Ensemble blending institucional
            blend_weight = (
                0.35 * rolling_perf
                + 0.30 * bayesian_mean
                + 0.20 * float(np.clip(obs.confidence, 0.0, 1.0))
                + 0.15 * float(np.clip(obs.historical_precision, 0.0, 1.0))
            )
            blend_weight *= float(np.clip(obs.regime_weight, 0.2, 2.0))
            blend_weight *= float(np.clip(obs.volatility_adjustment, 0.1, 1.5))
            if meta_weights is not None:
                blend_weight *= float(np.clip(meta_weights.get(obs.name, 0.5), 0.05, 5.0))
            blend_weight *= float(np.clip(meta_confidence, 0.1, 1.5))
            weight_components.append(max(blend_weight, 1e-6))
            names.append(obs.name)

        weights_np = np.asarray(weight_components)
        weights_np /= weights_np.sum()
        final_score = float(np.dot(weights_np, norm_scores))
        probability_up = self._sigmoid(final_score)
        calibrated_probability = self._sigmoid(self._platt_a * final_score + self._platt_b)

        return FusionResult(
            final_score=final_score,
            probability_up=probability_up,
            calibrated_probability=calibrated_probability,
            weights={name: float(w) for name, w in zip(names, weights_np)},
        )
