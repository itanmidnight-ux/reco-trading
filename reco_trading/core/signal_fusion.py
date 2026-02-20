from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.special import expit


@dataclass(slots=True)
class ModelDiagnostics:
    weight: float
    sharpe: float
    rolling_mean_pnl: float
    rolling_std_pnl: float


class SignalFusionEngine:
    """Institutional probabilistic signal fusion engine.

    Features:
    - Rolling performance memory and adaptive Sharpe-based weighting.
    - Bayesian shrinkage to avoid unstable weights in low-sample regimes.
    - Regime-aware weight boosts (trend/range/volatile).
    - Volatility dampening + confidence gating.
    - Optional online score calibration via logistic fit on recent labels.
    """

    def __init__(self, lookback: int = 256, min_samples: int = 20, epsilon: float = 1e-8) -> None:
        self.lookback = max(lookback, 32)
        self.min_samples = max(min_samples, 10)
        self.epsilon = epsilon

        self.model_weights: Dict[str, float] = {}
        self.performance_memory: Dict[str, deque[float]] = {}
        self.diagnostics: Dict[str, ModelDiagnostics] = {}

        # Online calibration state
        self._calibration_scores: deque[float] = deque(maxlen=self.lookback)
        self._calibration_labels: deque[int] = deque(maxlen=self.lookback)
        self._calibration_a: float = 1.0
        self._calibration_b: float = 0.0

    def update_performance(self, model_name: str, pnl: float) -> None:
        if model_name not in self.performance_memory:
            self.performance_memory[model_name] = deque(maxlen=self.lookback)
        self.performance_memory[model_name].append(float(pnl))

    def update_calibration(self, raw_score: float, label: int) -> None:
        self._calibration_scores.append(float(raw_score))
        self._calibration_labels.append(1 if label else 0)

    def _fit_online_calibration(self, lr: float = 0.02, epochs: int = 80) -> None:
        if len(self._calibration_scores) < self.min_samples:
            return

        x = np.asarray(self._calibration_scores, dtype=float)
        y = np.asarray(self._calibration_labels, dtype=float)

        a, b = self._calibration_a, self._calibration_b
        for _ in range(epochs):
            logits = np.clip(a * x + b, -35.0, 35.0)
            p = expit(logits)
            grad_a = float(np.mean((p - y) * x))
            grad_b = float(np.mean(p - y))
            a -= lr * grad_a
            b -= lr * grad_b

        self._calibration_a = float(a)
        self._calibration_b = float(b)

    def compute_dynamic_weights(self) -> Dict[str, float]:
        if not self.performance_memory:
            self.model_weights = {}
            self.diagnostics = {}
            return {}

        weights: Dict[str, float] = {}
        diagnostics: Dict[str, ModelDiagnostics] = {}

        for model, pnl_series in self.performance_memory.items():
            arr = np.asarray(pnl_series, dtype=float)
            if arr.size == 0:
                sharpe = 0.0
                mu = 0.0
                sigma = 1.0
                raw_weight = 1.0
            else:
                # Exponential recency weighting (institutional rolling behaviour)
                decay = np.exp(np.linspace(-2.0, 0.0, arr.size))
                decay /= decay.sum()
                mu = float(np.sum(arr * decay))
                sigma = float(np.sqrt(np.sum(decay * (arr - mu) ** 2)) + self.epsilon)
                sharpe = mu / sigma

                # Bayesian shrinkage toward neutral prior for small samples
                sample_strength = min(arr.size / float(self.min_samples), 1.0)
                shrunk_sharpe = sample_strength * sharpe
                raw_weight = max(float(np.exp(np.clip(shrunk_sharpe, -8.0, 8.0))), 0.01)

            weights[model] = raw_weight
            diagnostics[model] = ModelDiagnostics(
                weight=raw_weight,
                sharpe=float(sharpe),
                rolling_mean_pnl=float(mu),
                rolling_std_pnl=float(sigma),
            )

        total = sum(weights.values()) or 1.0
        normalized = {k: v / total for k, v in weights.items()}

        for model in diagnostics:
            diagnostics[model] = ModelDiagnostics(
                weight=float(normalized[model]),
                sharpe=diagnostics[model].sharpe,
                rolling_mean_pnl=diagnostics[model].rolling_mean_pnl,
                rolling_std_pnl=diagnostics[model].rolling_std_pnl,
            )

        self.model_weights = normalized
        self.diagnostics = diagnostics
        return normalized

    @staticmethod
    def _regime_multiplier(model: str, regime: str) -> float:
        regime_map: dict[str, dict[str, float]] = {
            'trend': {'momentum': 1.35, 'mean_reversion': 0.85, 'volatility_breakout': 1.15},
            'range': {'momentum': 0.85, 'mean_reversion': 1.35, 'volatility_breakout': 0.90},
            'volatile': {'momentum': 0.95, 'mean_reversion': 0.95, 'volatility_breakout': 1.40},
        }
        return regime_map.get(regime, {}).get(model, 1.0)

    def fuse(
        self,
        signals: Dict[str, float],
        regime: str,
        volatility: float,
        confidences: Dict[str, float] | None = None,
        calibrate: bool = True,
        transformer_prob_up: float | None = None,
    ) -> float:
        if not signals:
            return 0.5

        weights = self.compute_dynamic_weights()
        confidences = confidences or {}
        vol_adjustment = 1.0 / (1.0 + max(volatility, 0.0))

        adjusted_scores: list[float] = []
        weight_sum = 0.0

        if transformer_prob_up is not None:
            tp = float(np.clip(transformer_prob_up, 0.0, 1.0))
            signals = {**signals, "order_flow_transformer": 2.0 * tp - 1.0}
            confidences = {**confidences, "order_flow_transformer": max(abs(tp - 0.5) * 2.0, 0.25)}

        for model, score in signals.items():
            base_w = weights.get(model, 1.0 / max(len(signals), 1))
            regime_w = self._regime_multiplier(model, regime)
            confidence = float(np.clip(confidences.get(model, 1.0), 0.2, 1.5))
            w = base_w * regime_w * confidence
            adjusted_scores.append(w * float(score) * vol_adjustment)
            weight_sum += w

        final_score = float(np.sum(adjusted_scores) / max(weight_sum, self.epsilon))
        raw_probability = float(expit(final_score))

        if calibrate:
            self._fit_online_calibration()
            calibrated_probability = float(expit(self._calibration_a * final_score + self._calibration_b))
            return calibrated_probability

        return raw_probability
