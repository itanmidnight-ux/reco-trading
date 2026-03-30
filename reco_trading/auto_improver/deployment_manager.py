"""
Deployment Manager Module for Auto-Improver.
Manages safe deployment of new strategies with rollback capability.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from reco_trading.auto_improver.evaluator_engine import EvaluationResult
from reco_trading.auto_improver.strategy_generator import StrategyVariant

logger = logging.getLogger(__name__)


@dataclass
class DeploymentStatus:
    """Status of a deployment."""
    variant_id: str
    status: str
    deployed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    rollback_at: datetime | None = None
    reason: str = ""
    validation_runs: int = 0


class DeploymentManager:
    """Manages deployment of new strategies with safety measures."""

    def __init__(
        self,
        active_strategy_path: Path | None = None,
        backup_dir: Path | None = None,
    ):
        self.active_strategy_path = active_strategy_path or Path("./user_data/strategies/active.json")
        self.backup_dir = backup_dir or Path("./user_data/strategies/backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self._current_deployment: DeploymentStatus | None = None
        self._deployment_history: list[DeploymentStatus] = []
        
        self._current_variant: StrategyVariant | None = None
        self._previous_variant: StrategyVariant | None = None
        
        self._on_deploy_callback: Callable | None = None
        self._on_rollback_callback: Callable | None = None

    def set_deploy_callback(self, callback: Callable[[StrategyVariant], None]) -> None:
        """Set callback for deployment."""
        self._on_deploy_callback = callback

    def set_rollback_callback(self, callback: Callable[[StrategyVariant], None]) -> None:
        """Set callback for rollback."""
        self._on_rollback_callback = callback

    def can_deploy(
        self,
        new_variant: StrategyVariant,
        new_evaluation: EvaluationResult,
        current_evaluation: EvaluationResult | None = None,
    ) -> tuple[bool, str]:
        """Check if a variant can be safely deployed."""
        if new_evaluation.status != "success":
            return False, "Evaluation failed"
        
        if new_evaluation.metrics.get("total_trades", 0) < 10:
            return False, "Insufficient trades in backtest"
        
        if new_evaluation.metrics.get("roi", -100) < -50:
            return False, "ROI too negative"
        
        if new_evaluation.metrics.get("max_drawdown", 100) > 50:
            return False, "Maximum drawdown too high"
        
        if current_evaluation:
            new_roi = new_evaluation.metrics.get("roi", 0)
            current_roi = current_evaluation.metrics.get("roi", 0)
            
            if new_roi < current_roi:
                return False, f"New ROI ({new_roi:.2f}%) worse than current ({current_roi:.2f}%)"
        
        return True, "All checks passed"

    async def deploy(
        self,
        variant: StrategyVariant,
        evaluation: EvaluationResult,
        current_variant: StrategyVariant | None = None,
        current_evaluation: EvaluationResult | None = None,
    ) -> tuple[bool, str]:
        """Deploy a new strategy variant."""
        can_deploy, reason = self.can_deploy(variant, evaluation, current_evaluation)
        
        if not can_deploy:
            logger.warning(f"Deployment rejected: {reason}")
            return False, reason
        
        logger.info(f"Deploying strategy: {variant.name}")
        
        if current_variant:
            await self._backup_strategy(current_variant)
            self._previous_variant = current_variant
        
        await self._save_active_strategy(variant)
        
        self._current_variant = variant
        self._current_deployment = DeploymentStatus(
            variant_id=variant.id,
            status="deployed",
            reason=reason,
        )
        
        self._deployment_history.append(self._current_deployment)
        
        if self._on_deploy_callback:
            try:
                self._on_deploy_callback(variant)
            except Exception as e:
                logger.error(f"Deploy callback failed: {e}")
        
        logger.info(f"Successfully deployed strategy: {variant.name}")
        return True, f"Deployed: {variant.name}"

    async def rollback(self) -> tuple[bool, str]:
        """Rollback to previous strategy."""
        if not self._previous_variant:
            return False, "No previous strategy to rollback to"
        
        logger.warning(f"Rolling back to previous strategy: {self._previous_variant.name}")
        
        await self._save_active_strategy(self._previous_variant)
        
        if self._current_deployment:
            self._current_deployment.status = "rolled_back"
            self._current_deployment.rollback_at = datetime.now(timezone.utc)
        
        temp = self._current_variant
        self._current_variant = self._previous_variant
        self._previous_variant = temp
        
        if self._on_rollback_callback:
            try:
                self._on_rollback_callback(self._current_variant)
            except Exception as e:
                logger.error(f"Rollback callback failed: {e}")
        
        logger.info(f"Rolled back to strategy: {self._current_variant.name}")
        return True, f"Rolled back to: {self._current_variant.name}"

    async def validate_deployment(self, validation_results: dict[str, Any]) -> bool:
        """Validate a deployed strategy with real-time data."""
        if not self._current_deployment:
            return False
        
        self._current_deployment.validation_runs += 1
        
        is_healthy = validation_results.get("healthy", False)
        pnl = validation_results.get("pnl_today", 0)
        
        if not is_healthy:
            logger.warning("Deployment validation failed: unhealthy")
            return False
        
        if pnl < -5:
            logger.warning(f"Deployment validation failed: PnL {pnl}% too negative")
            return False
        
        logger.info(f"Deployment validation passed (run #{self._current_deployment.validation_runs})")
        return True

    async def _backup_strategy(self, variant: StrategyVariant) -> None:
        """Backup current strategy."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"{variant.id}_{timestamp}.json"
        
        with open(backup_file, "w") as f:
            json.dump(variant.to_dict(), f, indent=2)
        
        logger.info(f"Strategy backed up to {backup_file}")
        
        self._cleanup_old_backups(keep=10)

    async def _save_active_strategy(self, variant: StrategyVariant) -> None:
        """Save active strategy to file."""
        self.active_strategy_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.active_strategy_path, "w") as f:
            json.dump({
                "variant": variant.to_dict(),
                "deployed_at": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)
        
        logger.info(f"Active strategy saved to {self.active_strategy_path}")

    def _cleanup_old_backups(self, keep: int = 10) -> None:
        """Clean up old backup files."""
        backups = sorted(
            self.backup_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        
        for old_backup in backups[keep:]:
            old_backup.unlink()
            logger.debug(f"Removed old backup: {old_backup.name}")

    def get_current_strategy(self) -> StrategyVariant | None:
        """Get currently active strategy."""
        if self._current_variant:
            return self._current_variant
        
        if self.active_strategy_path.exists():
            try:
                with open(self.active_strategy_path, "r") as f:
                    data = json.load(f)
                
                variant_data = data.get("variant", {})
                return StrategyVariant(
                    id=variant_data["id"],
                    name=variant_data["name"],
                    parameters=variant_data["parameters"],
                    base_strategy=variant_data["base_strategy"],
                    generation=variant_data["generation"],
                )
            except Exception as e:
                logger.error(f"Error loading active strategy: {e}")
        
        return None

    def get_deployment_history(self) -> list[DeploymentStatus]:
        """Get deployment history."""
        return self._deployment_history

    def get_status(self) -> dict[str, Any]:
        """Get current deployment status."""
        return {
            "current_strategy": self._current_variant.name if self._current_variant else None,
            "current_id": self._current_variant.id if self._current_variant else None,
            "previous_strategy": self._previous_variant.name if self._previous_variant else None,
            "deployment_count": len(self._deployment_history),
            "status": self._current_deployment.status if self._current_deployment else "none",
        }
