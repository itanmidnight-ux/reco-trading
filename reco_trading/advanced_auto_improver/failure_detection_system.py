"""
Failure Detection System Module.
Detects failures and triggers protective actions.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures."""
    CONSECUTIVE_LOSSES = "consecutive_losses"
    EXCESSIVE_DRAWDOWN = "excessive_drawdown"
    ERRATIC_BEHAVIOR = "erratic_behavior"
    SILENT_ERRORS = "silent_errors"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SYSTEM_ERROR = "system_error"


class FailureSeverity(Enum):
    """Severity of failure."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FailureEvent:
    """Detected failure event."""
    failure_type: FailureType
    severity: FailureSeverity
    detected_at: datetime
    details: dict[str, Any]
    action_taken: str = ""
    resolved: bool = False


class FailureDetector:
    """Detects various types of failures."""

    def __init__(
        self,
        max_consecutive_losses: int = 7,
        max_drawdown: float = 30.0,
        max_performance_drop: float = 0.3,
        error_threshold: int = 5,
    ):
        self.max_consecutive_losses = max_consecutive_losses
        self.max_drawdown = max_drawdown
        self.max_performance_drop = max_performance_drop
        self.error_threshold = error_threshold
        
        self._consecutive_losses = 0
        self._current_drawdown = 0.0
        self._error_count = 0
        self._failure_history: list[FailureEvent] = []
        self._recent_pnls: list[float] = []

    def check_trade_result(self, pnl: float, is_error: bool = False) -> Optional[FailureEvent]:
        """Check trade result for failures."""
        if is_error:
            self._error_count += 1
        else:
            self._error_count = 0
        
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
        
        self._recent_pnls.append(pnl)
        if len(self._recent_pnls) > 50:
            self._recent_pnls.pop(0)
        
        failure = self._detect_failures()
        
        if failure:
            self._failure_history.append(failure)
            logger.warning(f"Failure detected: {failure.failure_type.value} - {failure.severity.value}")
        
        return failure

    def update_drawdown(self, drawdown: float) -> None:
        """Update current drawdown."""
        self._current_drawdown = drawdown

    def _detect_failures(self) -> Optional[FailureEvent]:
        """Detect any failures."""
        if self._consecutive_losses >= self.max_consecutive_losses:
            return FailureEvent(
                failure_type=FailureType.CONSECUTIVE_LOSSES,
                severity=FailureSeverity.HIGH if self._consecutive_losses >= 10 else FailureSeverity.MEDIUM,
                detected_at=datetime.now(timezone.utc),
                details={
                    "consecutive_losses": self._consecutive_losses,
                    "threshold": self.max_consecutive_losses,
                },
            )
        
        if self._current_drawdown > self.max_drawdown:
            return FailureEvent(
                failure_type=FailureType.EXCESSIVE_DRAWDOWN,
                severity=FailureSeverity.CRITICAL if self._current_drawdown > self.max_drawdown * 1.5 else FailureSeverity.HIGH,
                detected_at=datetime.now(timezone.utc),
                details={
                    "drawdown": self._current_drawdown,
                    "threshold": self.max_drawdown,
                },
            )
        
        if self._error_count >= self.error_threshold:
            return FailureEvent(
                failure_type=FailureType.SILENT_ERRORS,
                severity=FailureSeverity.HIGH,
                detected_at=datetime.now(timezone.utc),
                details={
                    "error_count": self._error_count,
                    "threshold": self.error_threshold,
                },
            )
        
        if self._is_erratic_behavior():
            return FailureEvent(
                failure_type=FailureType.ERRATIC_BEHAVIOR,
                severity=FailureSeverity.MEDIUM,
                detected_at=datetime.now(timezone.utc),
                details={"reason": "High variance in recent trades"},
            )
        
        return None

    def _is_erratic_behavior(self) -> bool:
        """Detect erratic behavior."""
        if len(self._recent_pnls) < 20:
            return False
        
        import math
        pnls = self._recent_pnls[-20:]
        
        mean = sum(pnls) / len(pnls)
        variance = sum((p - mean) ** 2 for p in pnls) / len(pnls)
        std_dev = math.sqrt(variance)
        
        if mean == 0:
            return std_dev > 20
        
        cv = abs(std_dev / mean)
        return cv > 2

    def get_active_failures(self) -> list[FailureEvent]:
        """Get unresolved failures."""
        return [f for f in self._failure_history if not f.resolved]

    def resolve_failure(self, index: int) -> None:
        """Mark failure as resolved."""
        if 0 <= index < len(self._failure_history):
            self._failure_history[index].resolved = True

    def get_failure_summary(self) -> dict[str, Any]:
        """Get summary of failures."""
        active = self.get_active_failures()
        
        return {
            "active_failures": len(active),
            "total_failures": len(self._failure_history),
            "consecutive_losses": self._consecutive_losses,
            "current_drawdown": self._current_drawdown,
            "error_count": self._error_count,
            "failure_types": {
                ft.value: sum(1 for f in self._failure_history if f.failure_type == ft)
                for ft in FailureType
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary."""
        return self.get_failure_summary()


class FailureResponseManager:
    """Manages responses to detected failures."""

    def __init__(self, failure_detector: FailureDetector):
        self.failure_detector = failure_detector
        self._paused = False
        self._pause_reason = ""
        self._actions_taken: list[str] = []

    def should_pause(self) -> tuple[bool, str]:
        """Check if trading should be paused."""
        active_failures = self.failure_detector.get_active_failures()
        
        for failure in active_failures:
            if failure.severity == FailureSeverity.CRITICAL:
                return True, f"{failure.failure_type.value} - critical"
            if failure.severity == FailureSeverity.HIGH and failure.failure_type == FailureType.CONSECUTIVE_LOSSES:
                return True, f"{failure.failure_type.value} - high"
        
        return False, ""

    def determine_action(self, failure: FailureEvent) -> str:
        """Determine appropriate action for failure."""
        actions = {
            FailureType.CONSECUTIVE_LOSSES: [
                "Reduce position size by 50%",
                "Switch to conservative strategy",
                "Pause trading",
            ],
            FailureType.EXCESSIVE_DRAWDOWN: [
                "Stop all trading immediately",
                "Close all positions",
                "Rollback to previous strategy",
            ],
            FailureType.ERRATIC_BEHAVIOR: [
                "Review strategy logic",
                "Reduce complexity",
            ],
            FailureType.SILENT_ERRORS: [
                "Enable detailed logging",
                "Check system health",
            ],
            FailureType.PERFORMANCE_DEGRADATION: [
                "Retrain model",
                "Switch strategy",
            ],
        }
        
        severity_actions = {
            FailureSeverity.LOW: 0,
            FailureSeverity.MEDIUM: 1,
            FailureSeverity.HIGH: min(2, len(actions.get(failure.failure_type, [""]))),
            FailureSeverity.CRITICAL: -1,
        }
        
        action_list = actions.get(failure.failure_type, ["No action"])
        index = severity_actions.get(failure.severity, 0)
        
        if index == -1:
            return "EMERGENCY STOP - Full portfolio protection"
        
        return action_list[index] if index < len(action_list) else action_list[-1]

    def execute_response(self, failure: FailureEvent) -> dict[str, Any]:
        """Execute response to failure."""
        action = self.determine_action(failure)
        
        failure.action_taken = action
        self._actions_taken.append(action)
        
        logger.error(f"Executing failure response: {action}")
        
        response = {
            "action": action,
            "failure_type": failure.failure_type.value,
            "severity": failure.severity.value,
            "should_pause": failure.severity in (FailureSeverity.CRITICAL, FailureSeverity.HIGH),
            "should_rollback": failure.severity == FailureSeverity.CRITICAL,
        }
        
        if failure.severity == FailureSeverity.CRITICAL:
            self._paused = True
            self._pause_reason = failure.failure_type.value
        
        return response

    def resume_trading(self) -> None:
        """Resume trading after failure."""
        self._paused = False
        self._pause_reason = ""
        self.failure_detector._consecutive_losses = 0
        self.failure_detector._current_drawdown = 0.0
        logger.info("Trading resumed after failure recovery")

    def is_paused(self) -> tuple[bool, str]:
        """Check if trading is paused."""
        return self._paused, self._pause_reason
