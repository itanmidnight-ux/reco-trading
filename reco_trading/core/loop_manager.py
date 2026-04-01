"""
Loop Manager - Orchestrates the complete auto-improvement cycle.
Coordinates SelfAnalyzer, IntelligentAutoImprover, and AutonomousOptimizer.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class LoopState(Enum):
    """States of the improvement loop."""
    IDLE = "idle"
    ANALYZING = "analyzing"
    OPTIMIZING = "optimizing"
    DEPLOYING = "deploying"
    MONITORING = "monitoring"
    PAUSED = "paused"


class SystemStatus(Enum):
    """Overall system status."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    RECOVERING = "recovering"


@dataclass
class LoopStatus:
    """Status of the improvement loop."""
    state: LoopState
    timestamp: datetime
    metrics: dict[str, Any] = field(default_factory=dict)
    active_systems: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


class LoopManager:
    """
    Master controller for all auto-improvement systems.
    Coordinates: SelfAnalyzer → IntelligentAutoImprover → AutonomousOptimizer
    """

    def __init__(self, enabled: bool = True):
        self.logger = logging.getLogger(__name__)
        self.enabled = enabled

        self._state = LoopState.IDLE
        self._system_status = SystemStatus.INITIALIZING

        self._self_analyzer = None
        self._auto_improver = None
        self._autonomous_optimizer = None

        self._loop_task: asyncio.Task | None = None
        self._check_interval_seconds = 60

        self._integration_callbacks: dict[str, callable] = {}

        self.logger.info("LoopManager initialized")

    async def initialize(
        self,
        auto_improver: Any = None,
        self_analyzer: Any = None,
        autonomous_optimizer: Any = None,
    ) -> None:
        """Initialize all subsystems."""
        self.logger.info("Initializing LoopManager subsystems")

        self._auto_improver = auto_improver
        self._self_analyzer = self_analyzer
        self._autonomous_optimizer = autonomous_optimizer

        # Set up callbacks for integration
        if self._autonomous_optimizer and self._auto_improver:
            self._autonomous_optimizer.set_callbacks(
                on_deployment=self._on_deployment,
                on_rollback=self._on_rollback,
                get_current_params=self._get_params_for_optimizer,
            )

        self._system_status = SystemStatus.RUNNING
        self.logger.info("LoopManager subsystems initialized")

    async def start(self) -> None:
        """Start the improvement loop."""
        if not self.enabled:
            self.logger.info("LoopManager disabled")
            return

        self.logger.info("Starting improvement loop")
        self._state = LoopState.ANALYZING
        self._loop_task = asyncio.create_task(self._main_loop())

    async def stop(self) -> None:
        """Stop the improvement loop."""
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass

        if self._self_analyzer:
            await self._self_analyzer.stop()
        if self._auto_improver:
            await self._auto_improver.stop()
        if self._autonomous_optimizer:
            await self._autonomous_optimizer.stop()

        self._state = LoopState.IDLE
        self.logger.info("LoopManager stopped")

    async def _main_loop(self) -> None:
        """Main orchestration loop."""
        while True:
            try:
                await asyncio.sleep(self._check_interval_seconds)

                await self._check_and_orchestrate()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Main loop error: {e}")
                await asyncio.sleep(60)

    async def _check_and_orchestrate(self) -> None:
        """Check all systems and orchestrate improvements."""
        self._state = LoopState.ANALYZING

        # Step 1: Get health status from SelfAnalyzer
        health_status = {}
        if self._self_analyzer:
            try:
                health_status = await self._self_analyzer.run_analysis()
                self._update_system_status(health_status)
            except Exception as e:
                self.logger.warning(f"SelfAnalyzer error: {e}")

        # Step 2: Check if AutoImprover should run
        if self._auto_improver:
            try:
                metrics = self._auto_improver.get_improvement_metrics()
                if self._auto_improver._should_optimize():
                    self._state = LoopState.OPTIMIZING
                    await self._auto_improver._run_optimization()
            except Exception as e:
                self.logger.warning(f"AutoImprover error: {e}")

        # Step 3: Check if AutonomousOptimizer should run
        if self._autonomous_optimizer:
            try:
                if self._autonomous_optimizer._current_cycle is None:
                    self._state = LoopState.OPTIMIZING
                    await self._autonomous_optimizer._run_cycle()
            except Exception as e:
                self.logger.warning(f"AutonomousOptimizer error: {e}")

        # Step 4: Check for critical issues
        if self._system_status == SystemStatus.CRITICAL:
            await self._handle_critical_state()

        self._state = LoopState.MONITORING

    def _update_system_status(self, health_status: dict[str, Any]) -> None:
        """Update system status based on health."""
        score = health_status.get("score", 100)

        if score >= 90:
            self._system_status = SystemStatus.RUNNING
        elif score >= 70:
            self._system_status = SystemStatus.RUNNING
        elif score >= 50:
            self._system_status = SystemStatus.DEGRADED
        else:
            self._system_status = SystemStatus.CRITICAL

    async def _handle_critical_state(self) -> None:
        """Handle critical system state."""
        self.logger.critical("System in CRITICAL state, initiating recovery")

        # Pause autonomous optimizer
        if self._autonomous_optimizer:
            await self._autonomous_optimizer.stop()

        # Increase auto-improvement frequency
        if self._auto_improver:
            self._auto_improver._optimization_interval_hours = 1

        # Run immediate analysis
        if self._self_analyzer:
            await self._self_analyzer.run_analysis()

        self._system_status = SystemStatus.RECOVERING

    def _on_deployment(self, hypothesis: Any, new_params: dict[str, Any]) -> None:
        """Handle successful deployment."""
        self.logger.info(f"Deployment successful: {hypothesis.id}")

        # Apply params to auto-improver
        if self._auto_improver:
            self._auto_improver._current_params = new_params

    def _on_rollback(self, hypothesis: Any) -> None:
        """Handle rollback."""
        self.logger.warning(f"Rollback executed: {hypothesis.id}")

    def _get_params_for_optimizer(self) -> dict[str, Any]:
        """Get current params for optimizer."""
        if self._auto_improver:
            return self._auto_improver.get_current_params()
        return {}

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "state": self._state.value,
            "system_status": self._system_status.value,
            "self_analyzer": {
                "enabled": self._self_analyzer.enabled if self._self_analyzer else False,
            } if self._self_analyzer else {},
            "auto_improver": {
                "enabled": self._auto_improver.enabled if self._auto_improver else False,
                "optimization_count": self._auto_improver._metrics.optimization_count if self._auto_improver else 0,
            } if self._auto_improver else {},
            "autonomous_optimizer": {
                "enabled": self._autonomous_optimizer.enabled if self._autonomous_optimizer else False,
                "active": self._autonomous_optimizer._current_cycle is not None if self._autonomous_optimizer else False,
            } if self._autonomous_optimizer else {},
        }

    async def force_optimization(self) -> dict[str, Any]:
        """Force an optimization cycle."""
        self.logger.info("Forcing optimization cycle")

        if self._auto_improver:
            await self._auto_improver._run_optimization()
            return {"success": True, "system": "auto_improver"}

        return {"success": False, "reason": "no_improver"}

    async def get_health_report(self) -> dict[str, Any]:
        """Get comprehensive health report."""
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "loop_state": self._state.value,
            "system_status": self._system_status.value,
        }

        if self._self_analyzer:
            try:
                report["health"] = await self._self_analyzer.run_analysis()
            except Exception as e:
                report["health"] = {"error": str(e)}

        if self._auto_improver:
            try:
                report["performance"] = self._auto_improver.get_performance_summary()
                report["improvements"] = self._auto_improver.get_improvement_metrics()
            except Exception as e:
                report["improver"] = {"error": str(e)}

        if self._autonomous_optimizer:
            try:
                report["optimizer"] = self._autonomous_optimizer.get_status()
            except Exception as e:
                report["optimizer"] = {"error": str(e)}

        return report