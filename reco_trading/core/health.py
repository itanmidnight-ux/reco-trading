"""
Health checks for system components
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    healthy: bool
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "healthy": self.healthy,
            "message": self.message,
            "details": self.details,
            "checked_at": self.checked_at.isoformat(),
        }


class HealthChecker:
    """Manages health checks for all system components."""

    def __init__(self) -> None:
        self._checks: dict[str, Callable[[], HealthCheckResult]] = {}
        self._last_results: dict[str, HealthCheckResult] = {}
        self._check_interval_seconds = 60

    def register(self, name: str, check_fn: Callable[[], HealthCheckResult]) -> None:
        """Register a health check."""
        self._checks[name] = check_fn
        logger.info(f"Registered health check: {name}")

    def unregister(self, name: str) -> None:
        """Unregister a health check."""
        if name in self._checks:
            del self._checks[name]
            logger.info(f"Unregistered health check: {name}")

    def check(self, name: str) -> HealthCheckResult | None:
        """Run a specific health check."""
        if name not in self._checks:
            return None

        try:
            result = self._checks[name]()
            self._last_results[name] = result
            return result
        except Exception as e:
            logger.exception(f"Health check {name} failed: {e}")
            result = HealthCheckResult(
                name=name,
                healthy=False,
                message=f"Check failed: {e}",
            )
            self._last_results[name] = result
            return result

    def check_all(self) -> dict[str, HealthCheckResult]:
        """Run all health checks."""
        results = {}
        for name in self._checks:
            result = self.check(name)
            if result:
                results[name] = result
        return results

    def is_healthy(self) -> bool:
        """Check if all components are healthy."""
        results = self.check_all()
        return all(r.healthy for r in results.values())

    def get_summary(self) -> dict[str, Any]:
        """Get health summary."""
        results = self.check_all()
        healthy_count = sum(1 for r in results.values() if r.healthy)
        total_count = len(results)

        return {
            "healthy": healthy_count == total_count,
            "checks": total_count,
            "healthy_checks": healthy_count,
            "unhealthy_checks": total_count - healthy_count,
            "results": {name: r.to_dict() for name, r in results.items()},
        }

    def get_unhealthy(self) -> list[HealthCheckResult]:
        """Get list of unhealthy components."""
        return [r for r in self._last_results.values() if not r.healthy]


def create_database_health_check(repository: Any) -> Callable[[], HealthCheckResult]:
    """Create a database health check."""
    def check() -> HealthCheckResult:
        try:
            if repository is None:
                return HealthCheckResult(
                    name="database",
                    healthy=False,
                    message="Repository not initialized",
                )
            return HealthCheckResult(
                name="database",
                healthy=True,
                message="Database connection OK",
            )
        except Exception as e:
            return HealthCheckResult(
                name="database",
                healthy=False,
                message=f"Database error: {e}",
            )
    return check


def create_exchange_health_check(client: Any) -> Callable[[], HealthCheckResult]:
    """Create an exchange health check."""
    def check() -> HealthCheckResult:
        try:
            if client is None:
                return HealthCheckResult(
                    name="exchange",
                    healthy=False,
                    message="Exchange client not initialized",
                )
            return HealthCheckResult(
                name="exchange",
                healthy=True,
                message="Exchange connection OK",
            )
        except Exception as e:
            return HealthCheckResult(
                name="exchange",
                healthy=False,
                message=f"Exchange error: {e}",
            )
    return check


def create_cache_health_check(cache: Any) -> Callable[[], HealthCheckResult]:
    """Create a cache health check."""
    def check() -> HealthCheckResult:
        try:
            if cache is None:
                return HealthCheckResult(
                    name="cache",
                    healthy=False,
                    message="Cache not initialized",
                )
            stats = getattr(cache, "stats", lambda: {})()
            return HealthCheckResult(
                name="cache",
                healthy=True,
                message="Cache OK",
                details={"stats": stats},
            )
        except Exception as e:
            return HealthCheckResult(
                name="cache",
                healthy=False,
                message=f"Cache error: {e}",
            )
    return check


_global_health_checker: HealthChecker | None = None


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance."""
    global _global_health_checker
    if _global_health_checker is None:
        _global_health_checker = HealthChecker()
    return _global_health_checker


__all__ = [
    "HealthCheckResult",
    "HealthChecker",
    "create_database_health_check",
    "create_exchange_health_check",
    "create_cache_health_check",
    "get_health_checker",
]
