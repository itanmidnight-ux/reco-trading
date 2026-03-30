"""
Self-Evaluation Engine Module.
System evaluates its own performance and triggers adaptations.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status."""
    EXCELLENT = "excellent"
    GOOD = "good"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class EvaluationResult:
    """Result of self-evaluation."""
    health_status: HealthStatus
    performance_score: float
    is_improving: bool
    should_retrain: bool
    should_switch_strategy: bool
    issues: list[str]
    recommendations: list[str]
    metrics: dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SelfEvaluationEngine:
    """System evaluates its own performance."""

    def __init__(
        self,
        window_size: int = 50,
        degradation_threshold: float = 0.2,
        retrain_threshold: float = 0.3,
    ):
        self.window_size = window_size
        self.degradation_threshold = degradation_threshold
        self.retrain_threshold = retrain_threshold
        
        self._performance_history: list[float] = []
        self._trade_history: list[dict[str, Any]] = []
        self._evaluation_history: list[EvaluationResult] = []

    def record_trade(self, trade: dict[str, Any]) -> None:
        """Record a trade for evaluation."""
        self._trade_history.append(trade)
        
        if len(self._trade_history) > self.window_size * 10:
            self._trade_history.pop(0)

    def evaluate(self) -> EvaluationResult:
        """Perform self-evaluation."""
        if len(self._trade_history) < 10:
            return self._insufficient_data_result()
        
        current_metrics = self._calculate_metrics()
        
        self._performance_history.append(current_metrics.get("roi", 0))
        
        if len(self._performance_history) > self.window_size:
            self._performance_history.pop(0)
        
        health_status = self._determine_health(current_metrics)
        is_improving = self._check_improving()
        should_retrain = self._check_should_retrain()
        should_switch = self._check_should_switch_strategy(current_metrics)
        
        issues = self._identify_issues(current_metrics)
        recommendations = self._generate_recommendations(
            health_status, is_improving, should_retrain, should_switch, issues
        )
        
        result = EvaluationResult(
            health_status=health_status,
            performance_score=current_metrics.get("roi", 0),
            is_improving=is_improving,
            should_retrain=should_retrain,
            should_switch_strategy=should_switch,
            issues=issues,
            recommendations=recommendations,
            metrics=current_metrics,
        )
        
        self._evaluation_history.append(result)
        
        if len(self._evaluation_history) > 100:
            self._evaluation_history.pop(0)
        
        logger.info(f"Self-evaluation: {health_status.value}, score={result.performance_score:.2f}")
        
        return result

    def _calculate_metrics(self) -> dict[str, Any]:
        """Calculate current performance metrics."""
        recent_trades = self._trade_history[-self.window_size:]
        
        if not recent_trades:
            return {"roi": 0, "win_rate": 0, "sharpe": 0, "drawdown": 0}
        
        pnls = [t.get("pnl", 0) for t in recent_trades]
        wins = sum(1 for p in pnls if p > 0)
        
        roi = sum(pnls)
        win_rate = wins / len(pnls) * 100 if pnls else 0
        
        returns = [p / 100 for p in pnls]
        sharpe = self._calculate_sharpe(returns) if len(returns) > 1 else 0
        
        drawdown = self._calculate_max_drawdown(pnls)
        
        return {
            "roi": roi,
            "win_rate": win_rate,
            "sharpe": sharpe,
            "drawdown": drawdown,
            "total_trades": len(pnls),
            "avg_pnl": sum(pnls) / len(pnls) if pnls else 0,
        }

    def _calculate_sharpe(self, returns: list[float]) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        import math
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance)
        
        if std_dev == 0:
            return 0.0
        
        return (mean_return / std_dev) * math.sqrt(252)

    def _calculate_max_drawdown(self, pnls: list[float]) -> float:
        """Calculate maximum drawdown."""
        equity = 100
        peak = equity
        max_dd = 0
        
        for pnl in pnls:
            equity += pnl
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        return max_dd

    def _determine_health(self, metrics: dict[str, Any]) -> HealthStatus:
        """Determine system health status."""
        roi = metrics.get("roi", 0)
        win_rate = metrics.get("win_rate", 0)
        sharpe = metrics.get("sharpe", 0)
        drawdown = metrics.get("drawdown", 0)
        
        if roi > 20 and win_rate > 60 and sharpe > 1.5 and drawdown < 10:
            return HealthStatus.EXCELLENT
        
        if roi > 10 and win_rate > 50 and sharpe > 1.0 and drawdown < 20:
            return HealthStatus.GOOD
        
        if roi > 0 and win_rate > 40 and sharpe > 0.5 and drawdown < 30:
            return HealthStatus.DEGRADED
        
        return HealthStatus.CRITICAL

    def _check_improving(self) -> bool:
        """Check if performance is improving."""
        if len(self._performance_history) < self.window_size:
            return True
        
        first_half = self._performance_history[:self.window_size // 2]
        second_half = self._performance_history[self.window_size // 2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        return second_avg > first_avg * (1 + self.degradation_threshold)

    def _check_should_retrain(self) -> bool:
        """Check if system should retrain."""
        if len(self._performance_history) < self.window_size:
            return False
        
        recent = self._performance_history[-self.window_size:]
        older = self._performance_history[:-self.window_size]
        
        if not older:
            return False
        
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        
        degradation = (older_avg - recent_avg) / abs(older_avg) if older_avg != 0 else 0
        
        return degradation > self.retrain_threshold

    def _check_should_switch_strategy(self, metrics: dict[str, Any]) -> bool:
        """Check if should switch to different strategy."""
        if metrics.get("drawdown", 0) > 40:
            return True
        
        if metrics.get("win_rate", 0) < 30:
            return True
        
        if metrics.get("sharpe", 0) < 0:
            return True
        
        return False

    def _identify_issues(self, metrics: dict[str, Any]) -> list[str]:
        """Identify current issues."""
        issues = []
        
        if metrics.get("drawdown", 0) > 30:
            issues.append(f"High drawdown: {metrics['drawdown']:.1f}%")
        
        if metrics.get("win_rate", 0) < 40:
            issues.append(f"Low win rate: {metrics['win_rate']:.1f}%")
        
        if metrics.get("sharpe", 0) < 0.5:
            issues.append(f"Low Sharpe ratio: {metrics['sharpe']:.2f}")
        
        if not self._check_improving():
            issues.append("Performance is declining")
        
        return issues

    def _generate_recommendations(
        self,
        health: HealthStatus,
        is_improving: bool,
        should_retrain: bool,
        should_switch: bool,
        issues: list[str],
    ) -> list[str]:
        """Generate recommendations."""
        recommendations = []
        
        if health == HealthStatus.CRITICAL:
            recommendations.append("CRITICAL: Consider pausing trading")
            recommendations.append("Review and potentially switch strategy")
        
        if should_retrain:
            recommendations.append("Trigger retraining with new data")
        
        if should_switch:
            recommendations.append("Consider switching to different strategy type")
        
        if not is_improving:
            recommendations.append("Performance not improving - analyze root cause")
        
        for issue in issues:
            if "drawdown" in issue.lower():
                recommendations.append("Reduce position sizes immediately")
            elif "win rate" in issue.lower():
                recommendations.append("Review entry criteria")
        
        if not recommendations:
            recommendations.append("System operating normally")
        
        return recommendations

    def _insufficient_data_result(self) -> EvaluationResult:
        """Result for insufficient data."""
        return EvaluationResult(
            health_status=HealthStatus.UNKNOWN,
            performance_score=0,
            is_improving=True,
            should_retrain=False,
            should_switch_strategy=False,
            issues=["Insufficient trading data for evaluation"],
            recommendations=["Continue trading to accumulate data"],
            metrics={},
        )

    def get_current_evaluation(self) -> Optional[EvaluationResult]:
        """Get current evaluation."""
        if not self._evaluation_history:
            return None
        return self._evaluation_history[-1]

    def get_health_trend(self, window: int = 5) -> str:
        """Get health trend over time."""
        if len(self._evaluation_history) < window:
            return "unknown"
        
        recent = self._evaluation_history[-window:]
        statuses = [e.health_status for e in recent]
        
        if all(s == HealthStatus.EXCELLENT for s in statuses):
            return "improving"
        if all(s in (HealthStatus.GOOD, HealthStatus.EXCELLENT) for s in statuses):
            return "stable"
        if any(s == HealthStatus.CRITICAL for s in statuses):
            return "declining"
        
        return "mixed"

    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary."""
        current = self.get_current_evaluation()
        
        return {
            "health_status": current.health_status.value if current else "unknown",
            "performance_score": current.performance_score if current else 0,
            "is_improving": current.is_improving if current else True,
            "should_retrain": current.should_retrain if current else False,
            "should_switch_strategy": current.should_switch_strategy if current else False,
            "issues": current.issues if current else [],
            "recommendations": current.recommendations if current else [],
            "health_trend": self.get_health_trend(),
        }
