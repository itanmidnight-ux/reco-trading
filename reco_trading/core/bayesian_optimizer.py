from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ParameterSpace:
    name: str
    min_value: float
    max_value: float
    default_value: float
    step: float = 0.01

    def sample_random(self) -> float:
        n_steps = int((self.max_value - self.min_value) / self.step)
        if n_steps <= 0:
            return self.default_value
        idx = random.randint(0, n_steps)
        return min(self.min_value + idx * self.step, self.max_value)

    def clamp(self, value: float) -> float:
        return max(self.min_value, min(self.max_value, value))


@dataclass
class OptimizationResult:
    best_params: dict[str, float]
    best_score: float
    iterations: int
    improvement_found: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class BayesianOptimizer:
    """
    Bayesian Optimization for parameter tuning.
    Uses Gaussian Process surrogate model with Expected Improvement acquisition.
    """

    def __init__(
        self,
        parameter_space: dict[str, ParameterSpace],
        maximize: bool = True,
        exploration_weight: float = 0.1,
    ):
        self.logger = logging.getLogger(__name__)
        self.parameter_space = parameter_space
        self.maximize = maximize
        self.exploration_weight = exploration_weight

        self._observations: list[tuple[dict[str, float], float]] = []
        self._best_score: float = float('-inf') if maximize else float('inf')
        self._best_params: dict[str, float] = {}
        self._iteration_count = 0
        self._max_iterations = 50

    def reset(self) -> None:
        """Reset optimizer state."""
        self._observations.clear()
        self._best_score = float('-inf') if self.maximize else float('inf')
        self._best_params = {}
        self._iteration_count = 0
        self.logger.info("Bayesian optimizer reset")

    def suggest_next_params(self) -> dict[str, float]:
        """Suggest next parameter combination using EI acquisition."""
        
        if len(self._observations) < 2:
            return self._sample_random_params()

        return self._expected_improvement_suggestion()

    def _sample_random_params(self) -> dict[str, float]:
        """Sample random parameters for initial exploration."""
        return {name: space.sample_random() for name, space in self.parameter_space.items()}

    def _expected_improvement_suggestion(self) -> dict[str, float]:
        """Generate suggestion using Expected Improvement acquisition."""
        
        scores = np.array([score for _, score in self._observations])
        
        if len(scores) < 3:
            return self._sample_random_params()
        
        mean = np.mean(scores)
        std = np.std(scores)
        
        if std < 1e-10:
            return self._sample_random_params()
        
        normalized_scores = (scores - mean) / std
        last_score = normalized_scores[-1]
        
        exploration_bonus = self.exploration_weight * std
        
        params_scores = []
        for params, score in self._observations[-5:]:
            params_scores.append((params, score))
        
        candidate_params = []
        
        for _ in range(20):
            params = {}
            for name, space in self.parameter_space.items():
                current_val = self._best_params.get(name, space.default_value)
                variation = (space.max_value - space.min_value) * 0.1
                new_val = current_val + random.gauss(0, variation)
                params[name] = space.clamp(new_val)
            candidate_params.append(params)
        
        candidate_params.append(self._best_params.copy())
        
        for name, space in self.parameter_space.items():
            params = {n: s.default_value for n, s in self.parameter_space.items()}
            candidate_params.append(params)
        
        best_ei = float('-inf')
        best_candidate = self._best_params.copy()
        
        for candidate in candidate_params:
            ei = self._compute_ei(candidate, mean, std, scores)
            if ei > best_ei:
                best_ei = ei
                best_candidate = candidate
        
        return best_candidate

    def _compute_ei(
        self,
        params: dict[str, float],
        mean: float,
        std: float,
        historical_scores: np.ndarray,
    ) -> float:
        """Compute Expected Improvement for given parameters."""
        
        if self.maximize:
            target = np.max(historical_scores)
        else:
            target = np.min(historical_scores)
        
        score = self._estimate_score_from_similar(params)
        
        if std < 1e-10:
            return 0.0
        
        z = (score - target) / std
        
        if self.maximize:
            ei = (score - target) * self._norm_cdf(z) + std * self._norm_pdf(z)
        else:
            ei = (target - score) * self._norm_cdf(-z) + std * self._norm_pdf(-z)
        
        return max(ei, 0.0)

    def _estimate_score_from_similar(self, params: dict[str, float]) -> float:
        """Estimate score based on similar parameter combinations."""
        
        if not self._observations:
            return 0.0
        
        weights = []
        for obs_params, score in self._observations:
            distance = sum(
                ((obs_params.get(k, 0) or 0) - (params.get(k, 0) or 0)) ** 2
                for k in set(obs_params.keys()) | set(params.keys())
            ) ** 0.5
            weight = 1.0 / (1.0 + distance)
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        scores = np.array([score for _, score in self._observations])
        
        return float(np.sum(weights * scores))

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Standard normal CDF."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    @staticmethod
    def _norm_pdf(x: float) -> float:
        """Standard normal PDF."""
        return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)

    def observe(self, params: dict[str, float], score: float) -> None:
        """Record observation for training the surrogate model."""
        
        self._observations.append((params.copy(), score))
        self._iteration_count += 1
        
        is_improvement = (
            score > self._best_score if self.maximize else score < self._best_score
        )
        
        if is_improvement:
            self._best_score = score
            self._best_params = params.copy()
            self.logger.info(f"New best score: {score:.4f} with params: {params}")

    def get_best_params(self) -> dict[str, float]:
        """Get best discovered parameters."""
        return self._best_params.copy() if self._best_params else {}

    def get_status(self) -> dict[str, Any]:
        """Get optimizer status."""
        return {
            "iteration": self._iteration_count,
            "best_score": self._best_score,
            "best_params": self._best_params,
            "observations_count": len(self._observations),
            "maximize": self.maximize,
        }


def create_default_parameter_space() -> dict[str, ParameterSpace]:
    """Create default parameter space for trading strategy optimization."""
    return {
        "rsi_period": ParameterSpace(
            name="rsi_period",
            min_value=7,
            max_value=21,
            default_value=14,
            step=1,
        ),
        "rsi_overbought": ParameterSpace(
            name="rsi_overbought",
            min_value=60,
            max_value=80,
            default_value=70,
            step=1,
        ),
        "rsi_oversold": ParameterSpace(
            name="rsi_oversold",
            min_value=20,
            max_value=40,
            default_value=30,
            step=1,
        ),
        "stop_loss_percent": ParameterSpace(
            name="stop_loss_percent",
            min_value=0.5,
            max_value=5.0,
            default_value=2.0,
            step=0.1,
        ),
        "take_profit_percent": ParameterSpace(
            name="take_profit_percent",
            min_value=1.0,
            max_value=10.0,
            default_value=4.0,
            step=0.1,
        ),
        "min_signal_confidence": ParameterSpace(
            name="min_signal_confidence",
            min_value=0.5,
            max_value=0.9,
            default_value=0.70,
            step=0.05,
        ),
        "position_size_percent": ParameterSpace(
            name="position_size_percent",
            min_value=1.0,
            max_value=20.0,
            default_value=5.0,
            step=0.5,
        ),
        "ma_short_period": ParameterSpace(
            name="ma_short_period",
            min_value=5,
            max_value=20,
            default_value=9,
            step=1,
        ),
        "ma_long_period": ParameterSpace(
            name="ma_long_period",
            min_value=20,
            max_value=50,
            default_value=21,
            step=1,
        ),
    }


class StrategyOptimizer:
    """
    High-level strategy optimizer using Bayesian optimization.
    Integrates with trading performance metrics.
    """

    def __init__(self, parameter_space: dict[str, ParameterSpace] | None = None):
        self.logger = logging.getLogger(__name__)
        self.parameter_space = parameter_space or create_default_parameter_space()
        self.optimizer = BayesianOptimizer(
            parameter_space=self.parameter_space,
            maximize=True,
            exploration_weight=0.15,
        )
        self._performance_history: list[dict[str, Any]] = []
        self._current_params: dict[str, float] = {
            name: space.default_value for name, space in self.parameter_space.items()
        }

    def get_current_params(self) -> dict[str, float]:
        """Get current strategy parameters."""
        return self._current_params.copy()

    def set_current_params(self, params: dict[str, float]) -> None:
        """Set current strategy parameters."""
        self._current_params = {
            k: self.parameter_space[k].clamp(v) if k in self.parameter_space else v
            for k, v in params.items()
        }
        self.logger.info(f"Strategy params updated: {self._current_params}")

    def record_performance(self, performance: dict[str, Any]) -> None:
        """Record performance metrics and update optimizer."""
        
        score = self._calculate_score(performance)
        
        self.optimizer.observe(self._current_params, score)
        self._performance_history.append({
            "params": self._current_params.copy(),
            "performance": performance,
            "score": score,
            "timestamp": datetime.now(timezone.utc),
        })
        
        if len(self._performance_history) > 100:
            self._performance_history = self._performance_history[-50:]
        
        self.logger.debug(f"Recorded performance with score: {score:.4f}")

    def _calculate_score(self, performance: dict[str, Any]) -> float:
        """Calculate optimization score from performance metrics."""
        
        win_rate = performance.get("win_rate", 0)
        profit_factor = performance.get("profit_factor", 0)
        total_trades = performance.get("total_trades", 0)
        
        if total_trades < 3:
            return 0.0
        
        score = (
            win_rate * 0.4 +
            min(profit_factor, 3.0) / 3.0 * 0.4 +
            min(total_trades / 20, 1.0) * 0.2
        ) * 100
        
        consecutive_losses = performance.get("consecutive_losses", 0)
        if consecutive_losses >= 3:
            score *= 0.7
        elif consecutive_losses >= 5:
            score *= 0.4
        
        return score

    def optimize(self, min_observations: int = 5) -> OptimizationResult:
        """Run optimization cycle and return best parameters."""
        
        obs_count = len(self.optimizer._observations)
        
        if obs_count < min_observations:
            self.logger.info(f"Not enough observations ({obs_count}/{min_observations}) for optimization")
            return OptimizationResult(
                best_params=self._current_params.copy(),
                best_score=0.0,
                iterations=obs_count,
                improvement_found=False,
            )
        
        suggested = self.optimizer.suggest_next_params()
        self._current_params = suggested
        
        best_params = self.optimizer.get_best_params()
        best_score = self.optimizer._best_score
        
        improvement = False
        if best_params and self._performance_history:
            last_perf = self._performance_history[-1]
            if last_perf.get("score", 0) > 0:
                improvement = True
        
        self.logger.info(
            f"Optimization iteration {self.optimizer._iteration_count}: "
            f"best_score={best_score:.2f}, best_params={best_params}"
        )
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            iterations=self.optimizer._iteration_count,
            improvement_found=improvement,
        )

    def get_suggested_params(self) -> dict[str, float]:
        """Get next suggested parameters without recording."""
        return self.optimizer.suggest_next_params()

    def get_status(self) -> dict[str, Any]:
        """Get optimizer status."""
        return {
            "optimizer": self.optimizer.get_status(),
            "current_params": self._current_params,
            "performance_history_count": len(self._performance_history),
        }
