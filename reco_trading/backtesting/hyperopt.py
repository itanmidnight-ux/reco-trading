from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd

from reco_trading.backtesting.engine import BacktestEngine
from reco_trading.backtesting.performance_metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class OptimizerType(Enum):
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"


@dataclass
class HyperoptSpace:
    """Definition of hyperparameter search space."""
    name: str
    min_value: float
    max_value: float
    step: float = 0.0
    categories: list[Any] = field(default_factory=list)
    is_discrete: bool = False


@dataclass
class OptimizationResult:
    """Result of a single optimization trial."""
    params: dict[str, float]
    score: float
    metrics: PerformanceMetrics
    trial_id: int


@dataclass
class BestResult:
    """Best result found during optimization."""
    params: dict[str, float]
    score: float
    metrics: PerformanceMetrics
    trials: int
    elapsed_seconds: float


class HyperoptOptimizer:
    """Hyperparameter optimizer for trading strategies."""

    def __init__(
        self,
        backtest_engine: BacktestEngine,
        space: dict[str, HyperoptSpace],
        metric_to_optimize: str = "sharpe_ratio",
        maximize: bool = True,
    ) -> None:
        self.backtest_engine = backtest_engine
        self.space = space
        self.metric_to_optimize = metric_to_optimize
        self.maximize = maximize
        self.optimizer_type = OptimizerType.RANDOM_SEARCH
        self.trials: list[OptimizationResult] = []
        self.best_result: BestResult | None = None
        self.start_time: datetime | None = None

    def _sample_params(self) -> dict[str, float]:
        """Sample random parameters from search space."""
        params = {}
        for name, space in self.space.items():
            if space.is_discrete and space.categories:
                params[name] = random.choice(space.categories)
            elif space.step > 0:
                num_steps = int((space.max_value - space.min_value) / space.step)
                step_index = random.randint(0, num_steps)
                params[name] = space.min_value + step_index * space.step
            else:
                params[name] = random.uniform(space.min_value, space.max_value)
        return params

    def _get_metric_value(self, metrics: PerformanceMetrics) -> float:
        """Extract metric value from metrics object."""
        metric_map = {
            "sharpe_ratio": metrics.sharpe_ratio,
            "total_return": metrics.total_return,
            "win_rate": metrics.win_rate,
            "profit_factor": metrics.profit_factor,
            "expectancy": metrics.expectancy,
            "max_drawdown": metrics.max_drawdown,
            "net_return": metrics.net_return_after_costs,
        }
        value = metric_map.get(self.metric_to_optimize, metrics.sharpe_ratio)
        if self.metric_to_optimize == "max_drawdown":
            value = abs(value)
            if self.maximize:
                value = -value
        elif not self.maximize:
            value = -value
        return value

    def _apply_params(self, params: dict[str, float]) -> None:
        """Apply parameters to backtest engine."""
        if "risk_fraction" in params:
            self.backtest_engine.risk_fraction = params["risk_fraction"]
        if "maker_fee_rate" in params:
            self.backtest_engine.maker_fee_rate = params["maker_fee_rate"]
        if "taker_fee_rate" in params:
            self.backtest_engine.taker_fee_rate = params["taker_fee_rate"]

    def _run_trial(self, params: dict[str, float], trial_id: int) -> OptimizationResult:
        """Run a single trial."""
        self._apply_params(params)
        result = self.backtest_engine.run(
            self.backtest_engine._test_frame5,
            self.backtest_engine._test_frame15,
            params.get("confidence_threshold", 0.75),
        )
        score = self._get_metric_value(result.metrics)
        return OptimizationResult(
            params=params,
            score=score,
            metrics=result.metrics,
            trial_id=trial_id,
        )

    def optimize(
        self,
        frame5m: pd.DataFrame,
        frame15m: pd.DataFrame,
        max_trials: int = 100,
        timeout_seconds: float | None = None,
    ) -> BestResult:
        """Run optimization."""
        self.backtest_engine._test_frame5 = frame5m
        self.backtest_engine._test_frame15 = frame15m
        self.trials = []
        self.start_time = datetime.now()

        logger.info(f"Starting optimization with {max_trials} trials...")

        for trial_id in range(max_trials):
            if timeout_seconds and self.start_time:
                elapsed = (datetime.now() - self.start_time).total_seconds()
                if elapsed >= timeout_seconds:
                    logger.info(f"Timeout reached after {elapsed:.1f}s")
                    break

            params = self._sample_params()
            result = self._run_trial(params, trial_id)
            self.trials.append(result)

            if self.best_result is None or (
                self.maximize and result.score > self.best_result.score
            ) or (
                not self.maximize and result.score < self.best_result.score
            ):
                elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
                self.best_result = BestResult(
                    params=result.params,
                    score=result.score,
                    metrics=result.metrics,
                    trials=trial_id + 1,
                    elapsed_seconds=elapsed,
                )
                logger.info(
                    f"Trial {trial_id + 1}: {self.metric_to_optimize}={result.score:.4f}, "
                    f"Return={result.metrics.total_return:.2%}, "
                    f"WinRate={result.metrics.win_rate:.2%}"
                )

        return self.best_result

    def get_top_results(self, n: int = 10) -> list[OptimizationResult]:
        """Get top N results."""
        sorted_trials = sorted(self.trials, key=lambda x: x.score, reverse=self.maximize)
        return sorted_trials[:n]

    def get_optimization_history(self) -> list[dict[str, Any]]:
        """Get optimization history."""
        return [
            {
                "trial_id": t.trial_id,
                "score": t.score,
                "params": t.params,
            }
            for t in self.trials
        ]


class GridSearchOptimizer(HyperoptOptimizer):
    """Grid search optimizer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer_type = OptimizerType.GRID_SEARCH

    def _generate_grid(self) -> list[dict[str, float]]:
        """Generate full grid of parameter combinations."""
        grid_points = []
        
        def recurse(index: int, current: dict[str, float]):
            if index == len(self.space):
                grid_points.append(current.copy())
                return
            
            name = list(self.space.keys())[index]
            space = self.space[name]
            
            if space.is_discrete and space.categories:
                for cat in space.categories:
                    current[name] = cat
                    recurse(index + 1, current)
            elif space.step > 0:
                value = space.min_value
                while value <= space.max_value:
                    current[name] = value
                    recurse(index + 1, current)
                    value += space.step
            else:
                current[name] = space.min_value
                recurse(index + 1, current)
        
        recurse(0, {})
        random.shuffle(grid_points)
        return grid_points

    def optimize(
        self,
        frame5m: pd.DataFrame,
        frame15m: pd.DataFrame,
        max_trials: int | None = None,
        timeout_seconds: float | None = None,
    ) -> BestResult:
        """Run grid search optimization."""
        self.backtest_engine._test_frame5 = frame5m
        self.backtest_engine._test_frame15 = frame15m
        self.trials = []
        self.start_time = datetime.now()

        grid = self._generate_grid()
        if max_trials:
            grid = grid[:max_trials]

        logger.info(f"Starting grid search with {len(grid)} configurations...")

        for trial_id, params in enumerate(grid):
            if timeout_seconds and self.start_time:
                elapsed = (datetime.now() - self.start_time).total_seconds()
                if elapsed >= timeout_seconds:
                    logger.info(f"Timeout reached after {elapsed:.1f}s")
                    break

            result = self._run_trial(params, trial_id)
            self.trials.append(result)

            if self.best_result is None or (
                self.maximize and result.score > self.best_result.score
            ) or (
                not self.maximize and result.score < self.best_result.score
            ):
                elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
                self.best_result = BestResult(
                    params=result.params,
                    score=result.score,
                    metrics=result.metrics,
                    trials=trial_id + 1,
                    elapsed_seconds=elapsed,
                )
                logger.info(
                    f"Grid {trial_id + 1}/{len(grid)}: {self.metric_to_optimize}={result.score:.4f}"
                )

        return self.best_result


def create_default_space() -> dict[str, HyperoptSpace]:
    """Create default hyperparameter search space."""
    return {
        "risk_fraction": HyperoptSpace(
            name="risk_fraction",
            min_value=0.005,
            max_value=0.05,
            step=0.005,
        ),
        "confidence_threshold": HyperoptSpace(
            name="confidence_threshold",
            min_value=0.5,
            max_value=0.95,
            step=0.05,
        ),
        "maker_fee_rate": HyperoptSpace(
            name="maker_fee_rate",
            min_value=0.0001,
            max_value=0.0005,
            step=0.0001,
        ),
        "taker_fee_rate": HyperoptSpace(
            name="taker_fee_rate",
            min_value=0.0005,
            max_value=0.001,
            step=0.0001,
        ),
    }


__all__ = [
    "HyperoptOptimizer",
    "GridSearchOptimizer",
    "HyperoptSpace",
    "OptimizationResult",
    "BestResult",
    "OptimizerType",
    "create_default_space",
]
