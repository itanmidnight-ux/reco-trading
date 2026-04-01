"""
Self-Analyzer System for Reco-Trading.
Provides automatic analysis of system performance, health monitoring, and diagnostic reports.
"""

from __future__ import annotations

import asyncio
import logging
import psutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class HealthLevel(Enum):
    """Health status levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    DEGRADED = "degraded"
    CRITICAL = "critical"


class IssueSeverity(Enum):
    """Issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    disk_usage_percent: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0
    process_count: int = 0
    thread_count: int = 0


@dataclass
class PerformanceMetrics:
    """Trading performance metrics."""
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_duration: float = 0.0
    signal_accuracy: float = 0.0
    false_signal_rate: float = 0.0
    avg_latency_ms: float = 0.0
    order_fill_rate: float = 0.0
    error_rate: float = 0.0


@dataclass
class HealthReport:
    """Complete health report."""
    timestamp: datetime
    health_level: HealthLevel
    overall_score: float
    system_metrics: SystemMetrics
    performance_metrics: PerformanceMetrics
    issues: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class SystemResourceMonitor:
    """Monitors system resources (CPU, Memory, Disk)."""

    def __init__(self, warning_cpu: float = 80.0, warning_memory: float = 85.0):
        self.warning_cpu = warning_cpu
        self.warning_memory = warning_memory
        self._baseline_cpu: float | None = None
        self._baseline_memory: float | None = None

    def get_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            net = psutil.net_io_counters()
            proc = psutil.Process()

            return SystemMetrics(
                cpu_percent=cpu,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=disk.percent,
                network_sent_mb=net.bytes_sent / (1024 * 1024),
                network_recv_mb=net.bytes_recv / (1024 * 1024),
                process_count=len(psutil.pids()),
                thread_count=proc.num_threads(),
            )
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return SystemMetrics()

    def check_thresholds(self, metrics: SystemMetrics) -> list[dict[str, Any]]:
        """Check if metrics exceed warning thresholds."""
        issues = []

        if metrics.cpu_percent > self.warning_cpu:
            issues.append({
                "severity": IssueSeverity.WARNING.value,
                "component": "CPU",
                "message": f"CPU usage high: {metrics.cpu_percent:.1f}%",
                "value": metrics.cpu_percent,
                "threshold": self.warning_cpu,
            })

        if metrics.memory_percent > self.warning_memory:
            issues.append({
                "severity": IssueSeverity.WARNING.value,
                "component": "Memory",
                "message": f"Memory usage high: {metrics.memory_percent:.1f}%",
                "value": metrics.memory_percent,
                "threshold": self.warning_memory,
            })

        if metrics.disk_usage_percent > 90:
            issues.append({
                "severity": IssueSeverity.ERROR.value,
                "component": "Disk",
                "message": f"Disk space critical: {metrics.disk_usage_percent:.1f}%",
                "value": metrics.disk_usage_percent,
                "threshold": 90,
            })

        return issues


class TradingPerformanceAnalyzer:
    """Analyzes trading performance metrics."""

    def __init__(self):
        self._trade_history: list[dict[str, Any]] = []
        self._signal_history: list[dict[str, Any]] = []
        self._latency_history: list[float] = []
        self._error_history: list[dict[str, Any]] = []

    def add_trade(self, trade_data: dict[str, Any]) -> None:
        self._trade_history.append(trade_data)
        if len(self._trade_history) > 1000:
            self._trade_history = self._trade_history[-1000:]

    def add_signal(self, signal_data: dict[str, Any]) -> None:
        self._signal_history.append(signal_data)
        if len(self._signal_history) > 1000:
            self._signal_history = self._signal_history[-1000:]

    def add_latency(self, latency_ms: float) -> None:
        self._latency_history.append(latency_ms)
        if len(self._latency_history) > 100:
            self._latency_history = self._latency_history[-100:]

    def add_error(self, error_data: dict[str, Any]) -> None:
        self._error_history.append(error_data)
        if len(self._error_history) > 100:
            self._error_history = self._error_history[-100:]

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Calculate performance metrics from history."""
        metrics = PerformanceMetrics()

        if not self._trade_history:
            return metrics

        winning = sum(1 for t in self._trade_history if t.get("pnl", 0) > 0)
        total = len(self._trade_history)

        metrics.total_trades = total
        metrics.win_rate = (winning / total * 100) if total > 0 else 0

        total_pnl = sum(t.get("pnl", 0) for t in self._trade_history)
        wins = sum(t.get("pnl", 0) for t in self._trade_history if t.get("pnl", 0) > 0)
        losses = abs(sum(t.get("pnl", 0) for t in self._trade_history if t.get("pnl", 0) < 0))

        metrics.profit_factor = wins / losses if losses > 0 else 0

        if self._signal_history:
            correct_signals = sum(1 for s in self._signal_history if s.get("result") == "success")
            metrics.signal_accuracy = (correct_signals / len(self._signal_history) * 100) if self._signal_history else 0

        if self._latency_history:
            metrics.avg_latency_ms = sum(self._latency_history) / len(self._latency_history)

        if self._error_history:
            recent_errors = len(self._error_history[-100:])
            metrics.error_rate = recent_errors / 100

        return metrics

    def analyze_issues(self) -> list[dict[str, Any]]:
        """Analyze performance and identify issues."""
        issues = []
        metrics = self.get_performance_metrics()

        if metrics.win_rate < 30 and metrics.total_trades > 10:
            issues.append({
                "severity": IssueSeverity.CRITICAL.value,
                "component": "Trading",
                "message": f"Win rate critically low: {metrics.win_rate:.1f}%",
            })

        if metrics.profit_factor < 1.0 and metrics.total_trades > 10:
            issues.append({
                "severity": IssueSeverity.ERROR.value,
                "component": "Trading",
                "message": f"Profit factor below 1.0: {metrics.profit_factor:.2f}",
            })

        if metrics.avg_latency_ms > 1000:
            issues.append({
                "severity": IssueSeverity.WARNING.value,
                "component": "Latency",
                "message": f"High latency detected: {metrics.avg_latency_ms:.0f}ms",
            })

        if metrics.error_rate > 0.1:
            issues.append({
                "severity": IssueSeverity.ERROR.value,
                "component": "Errors",
                "message": f"High error rate: {metrics.error_rate*100:.1f}%",
            })

        return issues


class DiagnosticReportGenerator:
    """Generates diagnostic reports."""

    def generate_report(
        self,
        system_metrics: SystemMetrics,
        performance_metrics: PerformanceMetrics,
        issues: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate comprehensive diagnostic report."""
        score = self._calculate_health_score(system_metrics, performance_metrics, issues)
        health_level = self._determine_health_level(score)

        recommendations = self._generate_recommendations(
            system_metrics, performance_metrics, issues
        )

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "health_level": health_level.value,
            "score": score,
            "system": {
                "cpu": f"{system_metrics.cpu_percent:.1f}%",
                "memory": f"{system_metrics.memory_percent:.1f}%",
                "disk": f"{system_metrics.disk_usage_percent:.1f}%",
            },
            "trading": {
                "total_trades": performance_metrics.total_trades,
                "win_rate": f"{performance_metrics.win_rate:.1f}%",
                "profit_factor": f"{performance_metrics.profit_factor:.2f}",
            },
            "issues_count": len(issues),
            "recommendations": recommendations,
        }

    def _calculate_health_score(
        self,
        system: SystemMetrics,
        performance: PerformanceMetrics,
        issues: list[dict[str, Any]],
    ) -> float:
        """Calculate overall health score (0-100)."""
        score = 100.0

        score -= min(30, system.cpu_percent * 0.3)
        score -= min(30, system.memory_percent * 0.3)
        score -= min(10, system.disk_usage_percent * 0.1)

        if performance.win_rate > 0:
            score += min(20, (performance.win_rate - 50) * 0.2)

        for issue in issues:
            if issue.get("severity") == IssueSeverity.CRITICAL.value:
                score -= 15
            elif issue.get("severity") == IssueSeverity.ERROR.value:
                score -= 10
            elif issue.get("severity") == IssueSeverity.WARNING.value:
                score -= 5

        return max(0.0, min(100.0, score))

    def _determine_health_level(self, score: float) -> HealthLevel:
        """Determine health level from score."""
        if score >= 90:
            return HealthLevel.EXCELLENT
        elif score >= 75:
            return HealthLevel.GOOD
        elif score >= 50:
            return HealthLevel.FAIR
        elif score >= 25:
            return HealthLevel.DEGRADED
        else:
            return HealthLevel.CRITICAL

    def _generate_recommendations(
        self,
        system: SystemMetrics,
        performance: PerformanceMetrics,
        issues: list[dict[str, Any]],
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []

        for issue in issues:
            if issue.get("component") == "CPU":
                recommendations.append("Consider reducing computational load or upgrading resources")
            elif issue.get("component") == "Memory":
                recommendations.append("Review memory usage patterns and consider optimization")
            elif issue.get("component") == "Trading":
                recommendations.append("Review trading strategy parameters")
            elif issue.get("component") == "Latency":
                recommendations.append("Check network connectivity and exchange API latency")

        if system.cpu_percent > 90:
            recommendations.append("URGENT: System CPU at critical levels")

        if system.memory_percent > 90:
            recommendations.append("URGENT: System memory at critical levels")

        if performance.win_rate < 40:
            recommendations.append("Consider pausing trading to review strategy")

        if not recommendations:
            recommendations.append("System operating normally")

        return recommendations


class SelfAnalyzer:
    """
    Main Self-Analyzer class that orchestrates all analysis components.
    """

    def __init__(self, enabled: bool = True):
        self.logger = logging.getLogger(__name__)
        self.enabled = enabled

        self._system_monitor = SystemResourceMonitor()
        self._performance_analyzer = TradingPerformanceAnalyzer()
        self._report_generator = DiagnosticReportGenerator()

        self._health_history: list[HealthReport] = []
        self._max_history = 100

        self._analysis_task: asyncio.Task | None = None
        self._analysis_interval_seconds = 60

        self.logger.info("SelfAnalyzer initialized")

    async def start(self) -> None:
        """Start the self-analysis loop."""
        if not self.enabled:
            self.logger.info("Self-analyzer disabled")
            return

        self.logger.info("Starting self-analyzer loop")
        self._analysis_task = asyncio.create_task(self._analysis_loop())

    async def stop(self) -> None:
        """Stop the self-analysis loop."""
        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Self-analyzer stopped")

    async def _analysis_loop(self) -> None:
        """Main analysis loop."""
        while True:
            try:
                await asyncio.sleep(self._analysis_interval_seconds)
                await self.run_analysis()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Analysis loop error: {e}")
                await asyncio.sleep(60)

    async def run_analysis(self) -> dict[str, Any]:
        """Run complete analysis and return report."""
        system_metrics = self._system_monitor.get_metrics()
        performance_metrics = self._performance_analyzer.get_performance_metrics()

        system_issues = self._system_monitor.check_thresholds(system_metrics)
        performance_issues = self._performance_analyzer.analyze_issues()

        all_issues = system_issues + performance_issues

        report = self._report_generator.generate_report(
            system_metrics, performance_metrics, all_issues
        )

        health_report = HealthReport(
            timestamp=datetime.now(timezone.utc),
            health_level=HealthLevel(report["health_level"]),
            overall_score=report["score"],
            system_metrics=system_metrics,
            performance_metrics=performance_metrics,
            issues=all_issues,
            recommendations=report["recommendations"],
        )

        self._health_history.append(health_report)
        if len(self._health_history) > self._max_history:
            self._health_history = self._health_history[-self._max_history:]

        if health_report.health_level in [HealthLevel.DEGRADED, HealthLevel.CRITICAL]:
            self.logger.warning(f"Health {health_report.health_level.value}: score={health_report.overall_score:.1f}")

        return report

    def record_trade(self, trade_data: dict[str, Any]) -> None:
        """Record trade for analysis."""
        self._performance_analyzer.add_trade(trade_data)

    def record_signal(self, signal_data: dict[str, Any]) -> None:
        """Record signal for analysis."""
        self._performance_analyzer.add_signal(signal_data)

    def record_latency(self, latency_ms: float) -> None:
        """Record latency for analysis."""
        self._performance_analyzer.add_latency(latency_ms)

    def record_error(self, error_data: dict[str, Any]) -> None:
        """Record error for analysis."""
        self._performance_analyzer.add_error(error_data)

    def get_current_health(self) -> dict[str, Any]:
        """Get current health status."""
        system = self._system_monitor.get_metrics()
        performance = self._performance_analyzer.get_performance_metrics()
        issues = self._system_monitor.check_thresholds(system) + self._performance_analyzer.analyze_issues()

        return self._report_generator.generate_report(system, performance, issues)

    def get_historical_health(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get historical health reports."""
        return [
            {
                "timestamp": h.timestamp.isoformat(),
                "health_level": h.health_level.value,
                "score": h.overall_score,
                "issues_count": len(h.issues),
            }
            for h in self._health_history[-limit:]
        ]

    def get_system_metrics(self) -> dict[str, Any]:
        """Get current system metrics."""
        metrics = self._system_monitor.get_metrics()
        return {
            "cpu_percent": metrics.cpu_percent,
            "memory_percent": metrics.memory_percent,
            "memory_used_mb": metrics.memory_used_mb,
            "disk_usage_percent": metrics.disk_usage_percent,
            "thread_count": metrics.thread_count,
        }

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get current performance metrics."""
        return self._performance_analyzer.get_performance_metrics().__dict__