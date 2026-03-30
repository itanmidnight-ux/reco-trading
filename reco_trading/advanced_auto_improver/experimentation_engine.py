"""
Experimentation Engine Module.
Runs controlled experiments to test new strategies.
"""

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class Experiment:
    """A single experiment."""
    experiment_id: str
    name: str
    strategy_variant: dict[str, Any]
    status: str
    metrics: dict[str, float]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    winner: bool = False


class ExperimentationEngine:
    """Manages controlled experiments."""

    def __init__(
        self,
        max_parallel_experiments: int = 3,
        min_experiment_trades: int = 20,
        confidence_level: float = 0.95,
    ):
        self.max_parallel_experiments = max_parallel_experiments
        self.min_experiment_trades = min_experiment_trades
        self.confidence_level = confidence_level
        
        self._experiments: list[Experiment] = []
        self._active_experiments: list[Experiment] = []

    def create_experiment(
        self,
        name: str,
        strategy_variant: dict[str, Any],
    ) -> Experiment:
        """Create a new experiment."""
        if len(self._active_experiments) >= self.max_parallel_experiments:
            logger.warning("Max parallel experiments reached")
        
        experiment_id = f"exp_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"
        
        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            strategy_variant=strategy_variant,
            status="running",
            metrics={},
        )
        
        self._experiments.append(experiment)
        self._active_experiments.append(experiment)
        
        logger.info(f"Created experiment: {experiment_id} - {name}")
        
        return experiment

    def record_experiment_trade(
        self,
        experiment_id: str,
        pnl: float,
        is_win: bool,
    ) -> None:
        """Record a trade for an experiment."""
        for exp in self._active_experiments:
            if exp.experiment_id == experiment_id:
                if "pnls" not in exp.metrics:
                    exp.metrics["pnls"] = []
                    exp.metrics["wins"] = 0
                    exp.metrics["trades"] = 0
                
                exp.metrics["pnls"].append(pnl)
                exp.metrics["trades"] += 1
                if is_win:
                    exp.metrics["wins"] += 1
                
                if exp.metrics["trades"] >= self.min_experiment_trades:
                    self._complete_experiment(exp)
                
                break

    def _complete_experiment(self, experiment: Experiment) -> None:
        """Complete an experiment and calculate results."""
        pnls = experiment.metrics.get("pnls", [])
        
        if not pnls:
            experiment.status = "insufficient_data"
            return
        
        experiment.metrics["total_pnl"] = sum(pnls)
        experiment.metrics["avg_pnl"] = sum(pnls) / len(pnls)
        experiment.metrics["win_rate"] = (experiment.metrics["wins"] / experiment.metrics["trades"]) * 100
        experiment.metrics["roi"] = experiment.metrics["total_pnl"]
        
        experiment.status = "completed"
        experiment.completed_at = datetime.now(timezone.utc)
        
        if experiment in self._active_experiments:
            self._active_experiments.remove(experiment)
        
        logger.info(f"Completed experiment: {experiment.experiment_id}, ROI: {experiment.metrics.get('roi', 0):.2f}")

    def determine_winner(
        self,
        control_metrics: dict[str, float],
        experiment_metrics: dict[str, float],
    ) -> tuple[bool, str]:
        """Determine if experiment is winner over control."""
        control_roi = control_metrics.get("roi", 0)
        experiment_roi = experiment_metrics.get("roi", 0)
        
        control_trades = control_metrics.get("trades", 0)
        experiment_trades = experiment_metrics.get("trades", 0)
        
        if experiment_trades < self.min_experiment_trades:
            return False, "Insufficient trades"
        
        roi_improvement = experiment_roi - control_roi
        
        control_std = self._calculate_std(control_metrics.get("pnls", []))
        experiment_std = self._calculate_std(experiment_metrics.get("pnls", []))
        
        if experiment_std > control_std * 1.5:
            return False, "Higher variance - less stable"
        
        if roi_improvement < 5:
            return False, "Not significant improvement"
        
        return True, f"Winner: +{roi_improvement:.2f}% ROI improvement"

    def _calculate_std(self, values: list[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance ** 0.5

    def get_active_experiments(self) -> list[Experiment]:
        """Get all active experiments."""
        return self._active_experiments

    def get_completed_experiments(self) -> list[Experiment]:
        """Get all completed experiments."""
        return [e for e in self._experiments if e.status == "completed"]

    def get_winners(self) -> list[Experiment]:
        """Get winning experiments."""
        return [e for e in self._experiments if e.winner]

    def get_experiment_summary(self) -> dict[str, Any]:
        """Get summary of experiments."""
        completed = self.get_completed_experiments()
        winners = self.get_winners()
        
        return {
            "total_experiments": len(self._experiments),
            "active_experiments": len(self._active_experiments),
            "completed_experiments": len(completed),
            "winner_experiments": len(winners),
            "win_rate": len(winners) / len(completed) * 100 if completed else 0,
        }

    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary."""
        return {
            "summary": self.get_experiment_summary(),
            "active": [
                {
                    "id": e.experiment_id,
                    "name": e.name,
                    "status": e.status,
                    "trades": e.metrics.get("trades", 0),
                }
                for e in self._active_experiments
            ],
            "winners": [
                {
                    "id": e.experiment_id,
                    "name": e.name,
                    "roi": e.metrics.get("roi", 0),
                }
                for e in self.get_winners()
            ],
        }


class ABTestRunner:
    """Runs A/B tests between strategies."""

    def __init__(self, experimentation_engine: ExperimentationEngine):
        self.experimentation = experimentation_engine

    def start_ab_test(
        self,
        test_name: str,
        control_params: dict[str, Any],
        variant_params: dict[str, Any],
    ) -> tuple[Experiment, Experiment]:
        """Start an A/B test."""
        control = self.experimentation.create_experiment(
            name=f"{test_name}_control",
            strategy_variant={"type": "control", "params": control_params},
        )
        
        variant = self.experimentation.create_experiment(
            name=f"{test_name}_variant",
            strategy_variant={"type": "variant", "params": variant_params},
        )
        
        logger.info(f"Started A/B test: {test_name}")
        
        return control, variant

    def analyze_ab_test(
        self,
        control: Experiment,
        variant: Experiment,
    ) -> dict[str, Any]:
        """Analyze A/B test results."""
        if control.status != "completed" or variant.status != "completed":
            return {"status": "incomplete", "message": "Both experiments must be completed"}
        
        is_winner, reason = self.experimentation.determine_winner(
            control.metrics,
            variant.metrics,
        )
        
        if is_winner:
            variant.winner = True
        
        return {
            "test_name": control.name.replace("_control", ""),
            "control_metrics": control.metrics,
            "variant_metrics": variant.metrics,
            "winner": variant.experiment_id if is_winner else control.experiment_id,
            "is_winner": is_winner,
            "reason": reason,
            "recommendation": "Deploy variant" if is_winner else "Keep control",
        }
