"""
Strategy Selector Module for Auto-Improver.
Compares and selects the best performing strategies.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reco_trading.auto_improver.evaluator_engine import EvaluationResult
from reco_trading.auto_improver.strategy_generator import StrategyVariant

logger = logging.getLogger(__name__)


@dataclass
class SelectionResult:
    """Result of strategy selection."""
    selected_variant: StrategyVariant | None
    evaluation: EvaluationResult | None
    reason: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class StrategySelector:
    """Selects best strategies based on multiple criteria."""

    def __init__(
        self,
        min_win_rate: float = 40.0,
        min_sharpe_ratio: float = 0.5,
        max_drawdown: float = 30.0,
        min_trades: int = 10,
    ):
        self.min_win_rate = min_win_rate
        self.min_sharpe_ratio = min_sharpe_ratio
        self.max_drawdown = max_drawdown
        self.min_trades = min_trades
        
        self._selections: list[SelectionResult] = []

    def select_best(
        self,
        results: list[EvaluationResult],
        variants: dict[str, StrategyVariant],
    ) -> SelectionResult:
        """Select the best strategy from evaluation results."""
        if not results:
            return SelectionResult(
                selected_variant=None,
                evaluation=None,
                reason="No results to compare",
            )
        
        valid_results = self._filter_valid_results(results)
        
        if not valid_results:
            return SelectionResult(
                selected_variant=None,
                evaluation=None,
                reason="No strategies met minimum criteria",
            )
        
        scored_results = self._score_results(valid_results)
        
        best = max(scored_results, key=lambda x: x[1])
        best_result, score = best
        
        variant = variants.get(best_result.variant_id)
        
        reason = f"Selected with score {score:.2f}: ROI={best_result.metrics.get('roi', 0):.2f}%, WinRate={best_result.metrics.get('win_rate', 0):.2f}%, Sharpe={best_result.metrics.get('sharpe_ratio', 0):.2f}"
        
        selection = SelectionResult(
            selected_variant=variant,
            evaluation=best_result,
            reason=reason,
        )
        
        self._selections.append(selection)
        
        logger.info(f"Selected strategy: {best_result.variant_name} - {reason}")
        return selection

    def _filter_valid_results(self, results: list[EvaluationResult]) -> list[EvaluationResult]:
        """Filter results that meet minimum criteria."""
        valid = []
        
        for result in results:
            if result.status != "success":
                continue
            
            metrics = result.metrics
            
            if metrics.get("total_trades", 0) < self.min_trades:
                continue
            
            if metrics.get("win_rate", 0) < self.min_win_rate:
                continue
            
            if metrics.get("max_drawdown", 100) > self.max_drawdown:
                continue
            
            valid.append(result)
        
        logger.info(f"Filtered {len(valid)}/{len(results)} valid strategies")
        return valid

    def _score_results(self, results: list[EvaluationResult]) -> list[tuple[EvaluationResult, float]]:
        """Score results based on multiple metrics."""
        scored = []
        
        for result in results:
            metrics = result.metrics
            
            roi = metrics.get("roi", 0)
            win_rate = metrics.get("win_rate", 0)
            sharpe = metrics.get("sharpe_ratio", 0)
            profit_factor = metrics.get("profit_factor", 0)
            drawdown = metrics.get("max_drawdown", 100)
            
            score = (
                roi * 0.25 +
                win_rate * 0.20 +
                sharpe * 2.0 * 0.20 +
                profit_factor * 10 * 0.15 +
                (100 - drawdown) * 0.20
            )
            
            scored.append((result, score))
        
        return scored

    def compare_strategies(
        self,
        results: list[EvaluationResult],
        variants: dict[str, StrategyVariant],
    ) -> list[dict[str, Any]]:
        """Compare multiple strategies."""
        comparison = []
        
        for result in results:
            variant = variants.get(result.variant_id)
            
            comparison.append({
                "variant_id": result.variant_id,
                "variant_name": result.variant_name,
                "metrics": result.metrics,
                "status": result.status,
                "evaluated_at": result.evaluated_at.isoformat(),
            })
        
        comparison.sort(key=lambda x: x["metrics"].get("roi", 0), reverse=True)
        
        return comparison

    def is_significantly_better(
        self,
        new_result: EvaluationResult,
        current_result: EvaluationResult,
        improvement_threshold: float = 0.1,
    ) -> bool:
        """Check if new strategy is significantly better than current."""
        new_roi = new_result.metrics.get("roi", 0)
        current_roi = current_result.metrics.get("roi", 0)
        
        roi_improvement = (new_roi - current_roi) / abs(current_roi) if current_roi != 0 else float("inf")
        
        new_sharpe = new_result.metrics.get("sharpe_ratio", 0)
        current_sharpe = current_result.metrics.get("sharpe_ratio", 0)
        
        sharpe_improvement = new_sharpe - current_sharpe
        
        logger.info(f"Comparison: ROI improvement={roi_improvement*100:.1f}%, Sharpe improvement={sharpe_improvement:.2f}")
        
        return (
            roi_improvement >= improvement_threshold or
            sharpe_improvement >= 0.5 and new_roi >= current_roi
        )

    def get_selection_history(self) -> list[SelectionResult]:
        """Get selection history."""
        return self._selections

    def save_rankings(self, results: list[EvaluationResult], output_dir: Path) -> None:
        """Save strategy rankings to file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        file_path = output_dir / f"rankings_{timestamp}.json"
        
        rankings = [
            {
                "rank": i + 1,
                "variant_id": r.variant_id,
                "variant_name": r.variant_name,
                "metrics": r.metrics,
                "evaluated_at": r.evaluated_at.isoformat(),
            }
            for i, r in enumerate(sorted(results, key=lambda x: x.metrics.get("roi", 0), reverse=True))
        ]
        
        with open(file_path, "w") as f:
            json.dump({
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "rankings": rankings,
            }, f, indent=2)
        
        logger.info(f"Rankings saved to {file_path}")
