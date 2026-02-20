from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class MetaLearningOutput:
    model_weights: dict[str, float]
    confidence_score: float


class AdaptiveMetaLearner:
    """Meta-learning para reponderación dinámica por performance/régimen/riesgo."""

    def __init__(self, model_names: list[str], window: int = 200) -> None:
        self.model_names = model_names
        self.window = max(window, 50)
        self._performance: dict[str, deque[float]] = {name: deque(maxlen=self.window) for name in model_names}
        self._regime_bias: dict[str, dict[str, float]] = defaultdict(dict)
        self._hyperparams: dict[str, float] = {"temperature": 1.0}

    def register_observation(self, model_name: str, was_correct: bool, regime: str) -> None:
        if model_name not in self._performance:
            self._performance[model_name] = deque(maxlen=self.window)
        self._performance[model_name].append(1.0 if was_correct else 0.0)
        score = float(np.mean(self._performance[model_name]))
        self._regime_bias[model_name][regime] = score

    @staticmethod
    def _bayesian_model_average(samples: np.ndarray) -> float:
        n = samples.size
        if n == 0:
            return 0.5
        alpha = 1.0 + float(samples.sum())
        beta = 1.0 + float(n - samples.sum())
        return float(alpha / (alpha + beta))

    def _drawdown_penalty(self, drawdown: float) -> float:
        return float(np.clip(1.0 - 2.5 * max(drawdown, 0.0), 0.05, 1.0))

    def _volatility_penalty(self, volatility: float) -> float:
        return float(np.clip(1.25 - 4.0 * max(volatility, 0.0), 0.25, 1.25))

    def optimize(
        self,
        *,
        regime: str,
        volatility: float,
        drawdown: float,
        base_weights: dict[str, float] | None = None,
    ) -> MetaLearningOutput:
        base_weights = base_weights or {name: 1.0 / max(len(self.model_names), 1) for name in self.model_names}

        scores: dict[str, float] = {}
        for name in self.model_names:
            samples = np.asarray(self._performance.get(name, deque()), dtype=float)
            rolling_perf = float(samples.mean()) if samples.size else 0.5
            bayesian = self._bayesian_model_average(samples)
            regime_fit = self._regime_bias.get(name, {}).get(regime, 0.5)
            score = 0.45 * rolling_perf + 0.35 * bayesian + 0.20 * regime_fit
            score *= self._drawdown_penalty(drawdown)
            score *= self._volatility_penalty(volatility)
            score *= float(np.clip(base_weights.get(name, 0.0), 1e-4, 1.0))
            scores[name] = max(score, 1e-6)

        score_vec = np.asarray(list(scores.values()), dtype=float)
        temperature = float(np.clip(self._hyperparams["temperature"], 0.25, 2.5))
        scaled = np.exp((score_vec - score_vec.max()) / temperature)
        normalized = scaled / scaled.sum()

        self._hyperparams["temperature"] = float(np.clip(0.9 * temperature + 0.1 * (1.0 + drawdown + volatility), 0.25, 2.5))
        confidence = float(np.clip(normalized.max() * (1.0 - drawdown), 0.0, 1.0))
        return MetaLearningOutput(
            model_weights={name: float(w) for name, w in zip(scores.keys(), normalized)},
            confidence_score=confidence,
        )
