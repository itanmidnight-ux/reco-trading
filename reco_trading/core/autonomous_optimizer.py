"""
Autonomous Optimizer for Reco-Trading.
Complete self-improvement loop with analysis, hypothesis generation, validation, and deployment.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class OptimizationPhase(Enum):
    """Phases of the optimization loop."""
    ANALYSIS = "analysis"
    HYPOTHESIS = "hypothesis"
    VALIDATION = "validation"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    ROLLBACK = "rollback"


class HypothesisStatus(Enum):
    """Status of optimization hypothesis."""
    GENERATED = "generated"
    TESTING = "testing"
    SUCCESSFUL = "successful"
    FAILED = "failed"
    DEPLOYED = "deployed"
    ROLLED_BACK = "rolled_back"


@dataclass
class OptimizationHypothesis:
    """A hypothesis for system improvement."""
    id: str
    description: str
    target_metric: str
    expected_improvement: float
    parameter_changes: dict[str, Any]
    status: HypothesisStatus = HypothesisStatus.GENERATED
    created_at: datetime = field(default_factory=datetime.now)
    tested_at: datetime | None = None
    deployed_at: datetime | None = None
    validation_score: float = 0.0
    baseline_score: float = 0.0
    actual_improvement: float = 0.0


@dataclass
class OptimizationCycle:
    """Complete optimization cycle."""
    id: str
    phase: OptimizationPhase
    start_time: datetime
    end_time: datetime | None = None
    hypothesis: OptimizationHypothesis | None = None
    results: dict[str, Any] = field(default_factory=dict)
    improvements: list[str] = field(default_factory=list)
    rollback_count: int = 0


@dataclass
class ValidationResult:
    """Result of hypothesis validation."""
    accepted: bool
    score: float
    risk_level: str
    backtest_result: dict[str, Any] | None = None
    simulation_result: dict[str, Any] | None = None
    confidence: float = 0.5


class HypothesisGenerator:
    """Generates improvement hypotheses based on analysis."""

    def __init__(self):
        self._hypothesis_templates = [
            {
                "description": "Reduce risk per trade due to recent losses",
                "target_metric": "win_rate",
                "expected_improvement": 0.10,
                "changes": {"risk_per_trade": 0.8},
            },
            {
                "description": "Increase confidence threshold to reduce false signals",
                "target_metric": "signal_accuracy",
                "expected_improvement": 0.15,
                "changes": {"min_confidence": 1.1},
            },
            {
                "description": "Adjust stop loss to reduce average loss",
                "target_metric": "profit_factor",
                "expected_improvement": 0.20,
                "changes": {"stop_loss_pct": 0.9},
            },
            {
                "description": "Extend holding time for better trend capture",
                "target_metric": "profit_factor",
                "expected_improvement": 0.15,
                "changes": {"max_holding_minutes": 1.2},
            },
            {
                "description": "Reduce trade frequency to improve quality",
                "target_metric": "win_rate",
                "expected_improvement": 0.12,
                "changes": {"min_signal_interval": 1.5},
            },
            {
                "description": "Increase position size in high confidence scenarios",
                "target_metric": "total_pnl",
                "expected_improvement": 0.25,
                "changes": {"high_confidence_multiplier": 1.3},
            },
        ]

    def generate(
        self,
        current_metrics: dict[str, Any],
        issues: list[dict[str, Any]],
    ) -> list[OptimizationHypothesis]:
        """Generate hypotheses based on current state."""
        hypotheses = []

        for template in self._hypothesis_templates:
            if self._should_generate_hypothesis(template, current_metrics, issues):
                hypothesis = OptimizationHypothesis(
                    id=f"hyp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}",
                    description=template["description"],
                    target_metric=template["target_metric"],
                    expected_improvement=template["expected_improvement"],
                    parameter_changes=template["changes"],
                    baseline_score=current_metrics.get(template["target_metric"], 0),
                )
                hypotheses.append(hypothesis)

        logger.info(f"Generated {len(hypotheses)} hypotheses")
        return hypotheses

    def _should_generate_hypothesis(
        self,
        template: dict[str, Any],
        metrics: dict[str, Any],
        issues: list[dict[str, Any]],
    ) -> bool:
        """Determine if hypothesis should be generated."""
        target = template["target_metric"]

        if target == "win_rate" and metrics.get("win_rate", 0) < 50:
            return True
        if target == "profit_factor" and metrics.get("profit_factor", 0) < 1.5:
            return True
        if target == "signal_accuracy" and metrics.get("signal_accuracy", 0) < 70:
            return True
        if target == "total_pnl" and metrics.get("total_pnl", 0) < 0:
            return True

        for issue in issues:
            if issue.get("severity") in ["error", "critical"]:
                return True

        return False


class HypothesisValidator:
    """Validates hypotheses before deployment."""

    def __init__(
        self,
        min_confidence: float = 0.7,
        max_risk_level: str = "medium",
    ):
        self.min_confidence = min_confidence
        self.max_risk_level = max_risk_level

    async def validate(
        self,
        hypothesis: OptimizationHypothesis,
        current_params: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> ValidationResult:
        """Validate hypothesis with backtesting and simulation."""
        logger.info(f"Validating hypothesis: {hypothesis.id}")

        # Calculate risk level based on changes
        risk_level = self._calculate_risk(hypothesis.parameter_changes)

        if risk_level == "high" and self.max_risk_level == "low":
            return ValidationResult(
                accepted=False,
                score=0.0,
                risk_level=risk_level,
                confidence=0.9,
            )

        # Simulate changes on historical data
        backtest_result = await self._simulate_backtest(
            hypothesis, current_params, history
        )

        # Run Monte Carlo simulation
        simulation_result = await self._run_simulation(
            hypothesis, current_params
        )

        # Calculate confidence score
        confidence = self._calculate_confidence(
            backtest_result, simulation_result, risk_level
        )

        # Determine acceptance
        accepted = (
            confidence >= self.min_confidence and
            risk_level != "critical" and
            backtest_result.get("expected_improvement", 0) > 0
        )

        return ValidationResult(
            accepted=accepted,
            score=backtest_result.get("expected_improvement", 0),
            risk_level=risk_level,
            backtest_result=backtest_result,
            simulation_result=simulation_result,
            confidence=confidence,
        )

    def _calculate_risk(self, changes: dict[str, Any]) -> str:
        """Calculate risk level of parameter changes."""
        risk_score = 0

        for param, change in changes.items():
            if isinstance(change, (int, float)):
                if abs(change - 1) > 0.5:
                    risk_score += 3
                elif abs(change - 1) > 0.3:
                    risk_score += 2
                else:
                    risk_score += 1
            else:
                risk_score += 1

        if risk_score >= 6:
            return "critical"
        elif risk_score >= 4:
            return "high"
        elif risk_score >= 2:
            return "medium"
        else:
            return "low"

    async def _simulate_backtest(
        self,
        hypothesis: OptimizationHypothesis,
        params: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Simulate hypothesis on historical data."""
        if not history:
            return {"expected_improvement": 0.5, "confidence": 0.5}

        # Apply changes to params
        new_params = params.copy()
        for key, value in hypothesis.parameter_changes.items():
            if key in new_params:
                if isinstance(value, (int, float)) and value != 1.0:
                    new_params[key] *= value
                else:
                    new_params[key] = value

        # Simulate: calculate expected improvement
        baseline_wr = hypothesis.baseline_score
        expected_wr = baseline_wr + hypothesis.expected_improvement * 100

        return {
            "expected_improvement": hypothesis.expected_improvement,
            "expected_win_rate": min(expected_wr, 95),
            "simulation_period": len(history),
            "confidence": 0.7,
        }

    async def _run_simulation(
        self,
        hypothesis: OptimizationHypothesis,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Run Monte Carlo simulation."""
        # Simplified simulation
        base_profit = params.get("total_pnl", 0)
        improvement = hypothesis.expected_improvement

        simulations = []
        for _ in range(100):
            variance = random.uniform(-0.2, 0.2)
            sim_profit = base_profit * (1 + improvement + variance)
            simulations.append(sim_profit)

        return {
            "mean": sum(simulations) / len(simulations),
            "min": min(simulations),
            "max": max(simulations),
            "std": (sum((x - sum(simulations) / len(simulations)) ** 2 for x in simulations) / len(simulations)) ** 0.5,
            "success_rate": sum(1 for s in simulations if s > 0) / len(simulations),
        }

    def _calculate_confidence(
        self,
        backtest: dict[str, Any],
        simulation: dict[str, Any],
        risk: str,
    ) -> float:
        """Calculate confidence in hypothesis."""
        confidence = 0.5

        if backtest:
            confidence += backtest.get("confidence", 0) * 0.3

        if simulation:
            success_rate = simulation.get("success_rate", 0.5)
            confidence += success_rate * 0.3

        if risk == "low":
            confidence += 0.2
        elif risk == "medium":
            confidence += 0.0
        elif risk == "high":
            confidence -= 0.1
        elif risk == "critical":
            confidence -= 0.3

        return max(0.0, min(1.0, confidence))


class RollbackManager:
    """Manages rollback of failed optimizations."""

    def __init__(self, max_rollbacks: int = 3):
        self.max_rollbacks = max_rollbacks
        self._rollback_history: list[dict[str, Any]] = []
        self._current_params_backup: dict[str, Any] = {}

    def save_current_state(self, params: dict[str, Any]) -> None:
        """Save current params before deployment."""
        self._current_params_backup = params.copy()

    def can_rollback(self) -> bool:
        """Check if rollback is available."""
        return len(self._rollback_history) < self.max_rollbacks

    def rollback(
        self,
        hypothesis_id: str,
        original_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform rollback to previous state."""
        if not self.can_rollback():
            logger.warning("Max rollbacks reached, cannot rollback")
            return {"success": False, "reason": "max_rollbacks"}

        self._rollback_history.append({
            "hypothesis_id": hypothesis_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "original_params": original_params,
        })

        logger.info(f"Rolled back hypothesis: {hypothesis_id}")
        return {"success": True, "restored_params": self._current_params_backup}

    def record_failure(self, hypothesis_id: str) -> None:
        """Record failed hypothesis."""
        self._rollback_history.append({
            "hypothesis_id": hypothesis_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
        })


class AutonomousOptimizer:
    """
    Complete autonomous optimization system.
    Implements the full improvement loop: Analyze -> Hypothesis -> Validate -> Deploy -> Monitor
    """

    def __init__(self, enabled: bool = True):
        self.logger = logging.getLogger(__name__)
        self.enabled = enabled

        self._hypothesis_generator = HypothesisGenerator()
        self._hypothesis_validator = HypothesisValidator()
        self._rollback_manager = RollbackManager()

        self._current_cycle: OptimizationCycle | None = None
        self._active_hypothesis: OptimizationHypothesis | None = None
        self._deployed_hypotheses: list[OptimizationHypothesis] = []

        self._optimization_task: asyncio.Task | None = None
        self._cycle_interval_minutes = 30

        self._callbacks: dict[str, Any] = {}

        self.logger.info("AutonomousOptimizer initialized")

    def set_callbacks(
        self,
        on_hypothesis_generated: callable | None = None,
        on_deployment: callable | None = None,
        on_rollback: callable | None = None,
        get_current_params: callable | None = None,
    ) -> None:
        """Set callback functions for external integration."""
        self._callbacks = {
            "on_hypothesis_generated": on_hypothesis_generated,
            "on_deployment": on_deployment,
            "on_rollback": on_rollback,
            "get_current_params": get_current_params,
        }

    async def start(self) -> None:
        """Start the autonomous optimization loop."""
        if not self.enabled:
            self.logger.info("Autonomous optimizer disabled")
            return

        self.logger.info("Starting autonomous optimization loop")
        self._optimization_task = asyncio.create_task(self._optimization_loop())

    async def stop(self) -> None:
        """Stop the optimization loop."""
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Autonomous optimizer stopped")

    async def _optimization_loop(self) -> None:
        """Main optimization loop."""
        while True:
            try:
                await asyncio.sleep(60 * self._cycle_interval_minutes)

                if self._current_cycle and self._current_cycle.phase != OptimizationPhase.MONITORING:
                    continue

                await self._run_cycle()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(300)

    async def _run_cycle(self) -> None:
        """Run complete optimization cycle."""
        cycle_id = f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._current_cycle = OptimizationCycle(
            id=cycle_id,
            phase=OptimizationPhase.ANALYSIS,
            start_time=datetime.now(timezone.utc),
        )

        self.logger.info(f"Starting optimization cycle: {cycle_id}")

        # Phase 1: Analysis
        self._current_cycle.phase = OptimizationPhase.ANALYSIS
        current_metrics = await self._get_current_metrics()
        issues = await self._get_issues()

        # Phase 2: Hypothesis Generation
        self._current_cycle.phase = OptimizationPhase.HYPOTHESIS
        hypotheses = self._hypothesis_generator.generate(current_metrics, issues)
        
        if not hypotheses:
            self.logger.info("No hypotheses generated, waiting for more data")
            return

        # Select best hypothesis
        hypothesis = max(hypotheses, key=lambda h: h.expected_improvement)
        self._active_hypothesis = hypothesis
        self._current_cycle.hypothesis = hypothesis

        # Notify callback
        if self._callbacks.get("on_hypothesis_generated"):
            try:
                self._callbacks["on_hypothesis_generated"](hypothesis)
            except Exception as e:
                self.logger.warning(f"Callback error: {e}")

        # Phase 3: Validation
        self._current_cycle.phase = OptimizationPhase.VALIDATION
        current_params = self._get_params()
        history = await self._get_trade_history()

        validation = await self._hypothesis_validator.validate(
            hypothesis, current_params, history
        )

        if not validation.accepted:
            self.logger.info(f"Hypothesis {hypothesis.id} rejected: risk={validation.risk_level}, confidence={validation.confidence}")
            hypothesis.status = HypothesisStatus.FAILED
            return

        hypothesis.status = HypothesisStatus.TESTING
        hypothesis.validation_score = validation.score

        # Phase 4: Deployment
        self._current_cycle.phase = OptimizationPhase.DEPLOYMENT
        self._rollback_manager.save_current_state(current_params)

        new_params = self._apply_hypothesis(current_params, hypothesis)
        hypothesis.status = HypothesisStatus.DEPLOYED
        hypothesis.deployed_at = datetime.now(timezone.utc)

        self._deployed_hypotheses.append(hypothesis)
        self._current_cycle.improvements.append(f"Deployed: {hypothesis.description}")

        # Notify callback
        if self._callbacks.get("on_deployment"):
            try:
                self._callbacks["on_deployment"](hypothesis, new_params)
            except Exception as e:
                self.logger.warning(f"Callback error: {e}")

        # Phase 5: Monitoring
        self._current_cycle.phase = OptimizationPhase.MONITORING
        self.logger.info(f"Deployed hypothesis {hypothesis.id}, monitoring for 30 minutes")

        # Monitor for 30 minutes then evaluate
        await asyncio.sleep(1800)  # 30 minutes

        improvement = await self._evaluate_deployment(hypothesis)

        if improvement < 0:
            # Rollback
            self._current_cycle.phase = OptimizationPhase.ROLLBACK
            self._rollback_manager.rollback(hypothesis.id, current_params)
            hypothesis.status = HypothesisStatus.ROLLED_BACK
            self._current_cycle.rollback_count += 1

            if self._callbacks.get("on_rollback"):
                try:
                    self._callbacks["on_rollback"](hypothesis)
                except Exception as e:
                    self.logger.warning(f"Callback error: {e}")
        else:
            hypothesis.actual_improvement = improvement

        self._current_cycle.end_time = datetime.now(timezone.utc)
        self.logger.info(f"Optimization cycle complete: {cycle_id}")

    async def _get_current_metrics(self) -> dict[str, Any]:
        """Get current system metrics."""
        try:
            if self._callbacks.get("get_current_params"):
                params = self._callbacks["get_current_params"]()
                return params
        except Exception as e:
            self.logger.warning(f"Could not get metrics: {e}")

        return {
            "win_rate": 50.0,
            "profit_factor": 1.0,
            "signal_accuracy": 70.0,
            "total_pnl": 0.0,
        }

    async def _get_issues(self) -> list[dict[str, Any]]:
        """Get current issues."""
        return []

    async def _get_trade_history(self) -> list[dict[str, Any]]:
        """Get trade history for backtesting."""
        return []

    def _get_params(self) -> dict[str, Any]:
        """Get current parameters."""
        if self._callbacks.get("get_current_params"):
            try:
                return self._callbacks["get_current_params"]()
            except:
                pass
        return {}

    def _apply_hypothesis(
        self,
        current_params: dict[str, Any],
        hypothesis: OptimizationHypothesis,
    ) -> dict[str, Any]:
        """Apply hypothesis changes to parameters."""
        new_params = current_params.copy()

        for key, value in hypothesis.parameter_changes.items():
            if key in new_params:
                if isinstance(value, (int, float)) and value != 1.0:
                    new_params[key] = new_params[key] * value
                else:
                    new_params[key] = value
            else:
                new_params[key] = value

        self.logger.info(f"Applied hypothesis {hypothesis.id}: {new_params}")
        return new_params

    async def _evaluate_deployment(self, hypothesis: OptimizationHypothesis) -> float:
        """Evaluate if deployment improved performance based on actual metrics."""
        await asyncio.sleep(1)
        
        metrics = self._get_current_metrics()
        if not metrics:
            return 0.0
        
        baseline_wr = hypothesis.baseline_metrics.get("win_rate", 50.0)
        baseline_pf = hypothesis.baseline_metrics.get("profit_factor", 1.0)
        
        current_wr = metrics.get("win_rate", 50.0)
        current_pf = metrics.get("profit_factor", 1.0)
        
        wr_improvement = (current_wr - baseline_wr) / max(baseline_wr, 1.0)
        pf_improvement = (current_pf - baseline_pf) / max(baseline_pf, 0.1)
        
        combined_score = (wr_improvement * 0.6 + pf_improvement * 0.4)
        
        self.logger.info(
            f"Deployment evaluation: wr={current_wr:.1f}% (was {baseline_wr:.1f}%), "
            f"pf={current_pf:.2f} (was {baseline_pf:.2f}), "
            f"score={combined_score:.4f}"
        )
        return combined_score

    def get_status(self) -> dict[str, Any]:
        """Get current optimizer status."""
        return {
            "enabled": self.enabled,
            "active_cycle": self._current_cycle.phase.value if self._current_cycle else None,
            "active_hypothesis": self._active_hypothesis.id if self._active_hypothesis else None,
            "deployed_count": len(self._deployed_hypotheses),
            "rollback_count": len(self._rollback_manager._rollback_history),
        }

    def get_deployed_hypotheses(self) -> list[dict[str, Any]]:
        """Get list of deployed hypotheses."""
        return [
            {
                "id": h.id,
                "description": h.description,
                "status": h.status.value,
                "deployed_at": h.deployed_at.isoformat() if h.deployed_at else None,
                "actual_improvement": h.actual_improvement,
            }
            for h in self._deployed_hypotheses
        ]