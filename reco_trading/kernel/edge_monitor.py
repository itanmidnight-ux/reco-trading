from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import erf, sqrt

import numpy as np


@dataclass(slots=True)
class EdgeSnapshot:
    edge_confidence_score: float
    t_stat: float
    p_value: float
    bayesian_prob_edge_positive: float
    sprt_state: str
    expectancy_ci_low: float
    expectancy_ci_high: float


class EdgeMonitor:
    def __init__(
        self,
        window: int = 120,
        alpha: float = 0.05,
        beta: float = 0.10,
        bayes_prior_mean: float = 0.0,
        bayes_prior_var: float = 1.0,
    ) -> None:
        self.window = max(int(window), 20)
        self.alpha = float(np.clip(alpha, 1e-4, 0.49))
        self.beta = float(np.clip(beta, 1e-4, 0.49))
        self.returns: deque[float] = deque(maxlen=self.window)
        self.prior_mean = float(bayes_prior_mean)
        self.prior_precision = float(1.0 / max(bayes_prior_var, 1e-9))
        self.posterior_mean = self.prior_mean
        self.posterior_precision = self.prior_precision
        self.sprt_llr = 0.0

    @staticmethod
    def _normal_cdf(x: float) -> float:
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    def update(self, realized_net_return: float | None) -> EdgeSnapshot:
        value = float(realized_net_return) if realized_net_return is not None else 0.0
        if not np.isfinite(value):
            value = 0.0
        self.returns.append(value)

        arr = np.asarray(self.returns, dtype=float)
        n = int(arr.size)
        if n < 3:
            return EdgeSnapshot(0.5, 0.0, 1.0, 0.5, 'INSUFFICIENT_DATA', 0.0, 0.0)

        mean = float(arr.mean())
        std = float(arr.std(ddof=1) or 0.0)
        effective_n = float(max(n * (1.0 - min(0.5, abs(np.corrcoef(arr[:-1], arr[1:])[0, 1]) if n > 4 else 0.0)), 1.0))
        stderr = float(max(std / sqrt(max(effective_n, 1.0)), 1e-9))
        t_stat = float(np.clip(mean / stderr, -25.0, 25.0))
        p_value = float(np.clip(2.0 * (1.0 - self._normal_cdf(abs(t_stat))), 0.0, 1.0))
        ci_margin = 1.96 * stderr
        expectancy_ci_low = float(mean - ci_margin)
        expectancy_ci_high = float(mean + ci_margin)

        # Bayesian online update: Normal-Normal approximation.
        obs_var = float(max(std * std, 1e-8))
        obs_precision = 1.0 / obs_var
        self.posterior_precision += obs_precision
        self.posterior_mean = (
            ((self.posterior_precision - obs_precision) * self.posterior_mean) + (obs_precision * value)
        ) / max(self.posterior_precision, 1e-9)
        post_std = float(sqrt(1.0 / max(self.posterior_precision, 1e-9)))
        bayes_prob = float(np.clip(1.0 - self._normal_cdf((0.0 - self.posterior_mean) / max(post_std, 1e-9)), 0.0, 1.0))

        # SPRT: H0 edge >= 0, H1 edge < 0 with conservative negative drift alternative.
        delta = float(max(0.10 * std, 1e-5))
        mu0 = 0.0
        mu1 = -delta
        var = float(max(std * std, 1e-8))
        increment = ((mu1 - mu0) * (value - ((mu0 + mu1) / 2.0))) / var
        self.sprt_llr = float(np.clip(self.sprt_llr + increment, -250.0, 250.0))
        upper = float(np.log((1.0 - self.beta) / max(self.alpha, 1e-9)))
        lower = float(np.log(max(self.beta, 1e-9) / max(1.0 - self.alpha, 1e-9)))
        if self.sprt_llr <= lower:
            sprt_state = 'EDGE_HEALTHY_H0'
        elif self.sprt_llr >= upper:
            sprt_state = 'EDGE_DECAY_H1'
        else:
            sprt_state = 'INCONCLUSIVE'

        confidence = float(np.clip((0.45 * bayes_prob) + (0.35 * (1.0 - p_value)) + (0.20 * (1.0 if sprt_state != 'EDGE_DECAY_H1' else 0.0)), 0.0, 1.0))
        return EdgeSnapshot(confidence, t_stat, p_value, bayes_prob, sprt_state, expectancy_ci_low, expectancy_ci_high)
