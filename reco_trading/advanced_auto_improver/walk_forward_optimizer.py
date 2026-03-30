"""
Walk-Forward Optimization Module.
Implements rolling window validation for realistic testing.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WindowResult:
    """Result of a single window optimization."""
    window_id: int
    train_start: datetime
    train_end: datetime
    validation_start: datetime
    validation_end: datetime
    best_params: dict[str, Any]
    train_metrics: dict[str, float]
    validation_metrics: dict[str, float]
    is_valid: bool


@dataclass
class WalkForwardResult:
    """Result of walk-forward optimization."""
    total_windows: int
    valid_windows: int
    avg_train_roi: float
    avg_validation_roi: float
    window_results: list[WindowResult]
    overall_roi_change: float
    stability_score: float
    recommended_params: dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class WalkForwardOptimizer:
    """Implements walk-forward optimization."""

    def __init__(
        self,
        train_days: int = 60,
        validation_days: int = 14,
        step_days: int = 7,
        min_train_days: int = 30,
    ):
        self.train_days = train_days
        self.validation_days = validation_days
        self.step_days = step_days
        self.min_train_days = min_train_days

    def optimize(
        self,
        data: list[dict[str, Any]],
        param_space: dict[str, list[Any]],
        objective: str = "sharpe_ratio",
        evaluation_func: Optional[Callable] = None,
    ) -> WalkForwardResult:
        """Run walk-forward optimization."""
        logger.info(f"Starting walk-forward optimization: train={self.train_days}d, val={self.validation_days}d")
        
        if not data:
            return self._empty_result()
        
        data = sorted(data, key=lambda x: x.get("timestamp", datetime.min))
        
        total_days = (data[-1].get("timestamp", datetime.now()) - data[0].get("timestamp", datetime.min)).days
        
        if total_days < self.train_days + self.validation_days:
            logger.warning("Insufficient data for walk-forward optimization")
            return self._empty_result()
        
        windows = self._generate_windows(total_days)
        
        window_results: list[WindowResult] = []
        
        for i, window in enumerate(windows):
            logger.info(f"Processing window {i+1}/{len(windows)}")
            
            train_data = self._get_window_data(data, window["train_start"], window["train_end"])
            val_data = self._get_window_data(data, window["val_start"], window["val_end"])
            
            if len(train_data) < self.min_train_days:
                continue
            
            best_params = self._optimize_window(
                train_data,
                param_space,
                objective,
                evaluation_func,
            )
            
            train_metrics = self._evaluate_data(train_data, best_params, evaluation_func)
            val_metrics = self._evaluate_data(val_data, best_params, evaluation_func)
            
            is_valid = val_metrics.get("roi", -100) > -50
            
            window_result = WindowResult(
                window_id=i,
                train_start=window["train_start"],
                train_end=window["train_end"],
                validation_start=window["val_start"],
                validation_end=window["val_end"],
                best_params=best_params,
                train_metrics=train_metrics,
                validation_metrics=val_metrics,
                is_valid=is_valid,
            )
            
            window_results.append(window_result)
        
        return self._compile_results(window_results)

    def _generate_windows(self, total_days: int) -> list[dict[str, int]]:
        """Generate walk-forward windows."""
        windows = []
        
        current_train_end = total_days - self.validation_days
        
        while current_train_end >= self.train_days:
            train_start = current_train_end - self.train_days
            train_end = current_train_end
            
            val_start = train_end
            val_end = min(val_start + self.validation_days, total_days)
            
            windows.append({
                "train_start": train_start,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
            })
            
            current_train_end -= self.step_days
        
        windows.reverse()
        
        return windows

    def _get_window_data(
        self,
        data: list[dict[str, Any]],
        start_day: int,
        end_day: int,
    ) -> list[dict[str, Any]]:
        """Extract data for a window."""
        return data[start_day:end_day]

    def _optimize_window(
        self,
        train_data: list[dict[str, Any]],
        param_space: dict[str, list[Any]],
        objective: str,
        evaluation_func: Optional[Callable],
    ) -> dict[str, Any]:
        """Optimize parameters for a single window."""
        if evaluation_func:
            best_score = float("-inf")
            best_params = {}
            
            import itertools
            param_combinations = list(itertools.product(*param_space.values()))
            
            sample_size = min(50, len(param_combinations))
            import random
            sampled = random.sample(param_combinations, sample_size) if sample_size < len(param_combinations) else param_combinations
            
            for params in sampled:
                param_dict = dict(zip(param_space.keys(), params))
                
                score = evaluation_func(train_data, param_dict)
                
                if score > best_score:
                    best_score = score
                    best_params = param_dict
            
            return best_params
        
        return {k: v[0] for k, v in param_space.items()}

    def _evaluate_data(
        self,
        data: list[dict[str, Any]],
        params: dict[str, Any],
        evaluation_func: Optional[Callable],
    ) -> dict[str, float]:
        """Evaluate data with given parameters."""
        if evaluation_func and data:
            return evaluation_func(data, params)
        
        return {
            "roi": 0.0,
            "win_rate": 50.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0,
        }

    def _compile_results(self, window_results: list[WindowResult]) -> WalkForwardResult:
        """Compile results from all windows."""
        if not window_results:
            return self._empty_result()
        
        valid_results = [r for r in window_results if r.is_valid]
        
        if not valid_results:
            return WalkForwardResult(
                total_windows=len(window_results),
                valid_windows=0,
                avg_train_roi=0,
                avg_validation_roi=0,
                window_results=window_results,
                overall_roi_change=0,
                stability_score=0,
                recommended_params={},
            )
        
        train_rois = [r.train_metrics.get("roi", 0) for r in valid_results]
        val_rois = [r.validation_metrics.get("roi", 0) for r in valid_results]
        
        avg_train_roi = sum(train_rois) / len(train_rois)
        avg_val_roi = sum(val_rois) / len(val_rois)
        
        overall_roi_change = avg_val_roi - avg_train_roi
        
        stability_score = self._calculate_stability(val_rois)
        
        recommended_params = self._aggregate_params(valid_results)
        
        return WalkForwardResult(
            total_windows=len(window_results),
            valid_windows=len(valid_results),
            avg_train_roi=avg_train_roi,
            avg_validation_roi=avg_val_roi,
            window_results=window_results,
            overall_roi_change=overall_roi_change,
            stability_score=stability_score,
            recommended_params=recommended_params,
        )

    def _calculate_stability(self, values: list[float]) -> float:
        """Calculate stability score."""
        if len(values) < 2:
            return 0.5
        
        values_array = np.array(values)
        mean = np.mean(values_array)
        
        if mean == 0:
            return 0.0
        
        cv = np.std(values_array) / abs(mean)
        
        return max(0, 1 - min(cv, 1))

    def _aggregate_params(self, results: list[WindowResult]) -> dict[str, Any]:
        """Aggregate parameters from best windows."""
        best_results = sorted(results, key=lambda r: r.validation_metrics.get("roi", 0), reverse=True)
        
        if not best_results:
            return {}
        
        top_results = best_results[:max(1, len(best_results) // 2)]
        
        aggregated = {}
        
        for param_key in top_results[0].best_params.keys():
            values = [r.best_params.get(param_key) for r in top_results if param_key in r.best_params]
            
            if values:
                try:
                    numeric_values = [v for v in values if isinstance(v, (int, float))]
                    if numeric_values:
                        aggregated[param_key] = sum(numeric_values) / len(numeric_values)
                    else:
                        aggregated[param_key] = max(set(values), key=list(values).count)
                except:
                    aggregated[param_key] = values[0]
        
        return aggregated

    def _empty_result(self) -> WalkForwardResult:
        """Create empty result."""
        return WalkForwardResult(
            total_windows=0,
            valid_windows=0,
            avg_train_roi=0,
            avg_validation_roi=0,
            window_results=[],
            overall_roi_change=0,
            stability_score=0,
            recommended_params={},
        )

    def validate_robustness(
        self,
        walk_forward_result: WalkForwardResult,
    ) -> tuple[bool, str]:
        """Validate if result is robust enough for deployment."""
        if walk_forward_result.valid_windows < 3:
            return False, "Insufficient valid windows"
        
        if walk_forward_result.stability_score < 0.3:
            return False, f"Low stability: {walk_forward_result.stability_score:.2f}"
        
        if walk_forward_result.avg_validation_roi < 0:
            return False, "Negative validation ROI"
        
        if walk_forward_result.overall_roi_change < -30:
            return False, f"Large ROI degradation: {walk_forward_result.overall_roi_change:.1f}%"
        
        return True, "Robust enough for deployment"

    def to_dict(self) -> dict[str, Any]:
        """Export configuration to dictionary."""
        return {
            "train_days": self.train_days,
            "validation_days": self.validation_days,
            "step_days": self.step_days,
            "min_train_days": self.min_train_days,
        }
