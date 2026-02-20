from __future__ import annotations

import asyncio
import inspect
import statistics
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Deque, Literal, Sequence

from loguru import logger

Severity = Literal['warning', 'critical']


@dataclass(slots=True)
class HealthSnapshot:
    timestamp: float
    latency_avg_ms: float
    latency_p95_ms: float
    api_error_rate: float
    slippage_mean_bps: float
    spread_drift_bps: float
    intraday_drawdown_ratio: float
    gpu_memory_ratio: float
    redis_ok: bool
    postgresql_ok: bool


@dataclass(slots=True)
class MetricRule:
    metric: str
    warning_threshold: float
    critical_threshold: float
    window_seconds: float
    direction: Literal['gt', 'lt'] = 'gt'


@dataclass(slots=True)
class HealthAnomaly:
    metric: str
    severity: Severity
    value: float
    threshold: float
    window_seconds: float
    detail: str
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class HealthEvaluation:
    snapshot: HealthSnapshot
    anomalies: list[HealthAnomaly]
    actions: list[str]


DEFAULT_RULES: tuple[MetricRule, ...] = (
    MetricRule('latency_avg_ms', warning_threshold=80.0, critical_threshold=140.0, window_seconds=60.0),
    MetricRule('latency_p95_ms', warning_threshold=140.0, critical_threshold=250.0, window_seconds=60.0),
    MetricRule('api_error_rate', warning_threshold=0.02, critical_threshold=0.05, window_seconds=120.0),
    MetricRule('slippage_mean_bps', warning_threshold=6.0, critical_threshold=12.0, window_seconds=180.0),
    MetricRule('spread_drift_bps', warning_threshold=8.0, critical_threshold=15.0, window_seconds=120.0),
    MetricRule('intraday_drawdown_ratio', warning_threshold=0.05, critical_threshold=0.09, window_seconds=300.0),
    MetricRule('gpu_memory_ratio', warning_threshold=0.88, critical_threshold=0.95, window_seconds=90.0),
    MetricRule('redis_ok', warning_threshold=0.5, critical_threshold=0.5, window_seconds=30.0, direction='lt'),
    MetricRule('postgresql_ok', warning_threshold=0.5, critical_threshold=0.5, window_seconds=30.0, direction='lt'),
)


class SystemHealthMonitor:
    def __init__(
        self,
        *,
        rules: Sequence[MetricRule] | None = None,
        metrics: Any | None = None,
    ) -> None:
        self.rules = tuple(rules or DEFAULT_RULES)
        self.metrics = metrics
        self._series: dict[str, Deque[tuple[float, float]]] = defaultdict(deque)

    async def collect_snapshot(
        self,
        *,
        latencies_ms: Sequence[float],
        api_errors: int,
        api_requests: int,
        slippage_samples_bps: Sequence[float],
        spread_drift_samples_bps: Sequence[float],
        intraday_equity_curve: Sequence[float],
        gpu_memory_used_bytes: int | float,
        gpu_memory_total_bytes: int | float,
        redis_client: Any,
        database: Any,
    ) -> HealthSnapshot:
        redis_ok, _ = await self._check_redis(redis_client)
        pg_ok, _ = await self._check_postgresql(database)

        return HealthSnapshot(
            timestamp=time.time(),
            latency_avg_ms=_mean_or_zero(latencies_ms),
            latency_p95_ms=_p95_or_zero(latencies_ms),
            api_error_rate=(float(api_errors) / float(api_requests)) if api_requests > 0 else 0.0,
            slippage_mean_bps=_mean_or_zero(slippage_samples_bps),
            spread_drift_bps=_mean_or_zero(spread_drift_samples_bps),
            intraday_drawdown_ratio=_intraday_drawdown(intraday_equity_curve),
            gpu_memory_ratio=(float(gpu_memory_used_bytes) / float(gpu_memory_total_bytes)) if gpu_memory_total_bytes > 0 else 0.0,
            redis_ok=redis_ok,
            postgresql_ok=pg_ok,
        )

    async def evaluate(self, snapshot: HealthSnapshot, *, quant_kernel: Any | None = None) -> HealthEvaluation:
        now = snapshot.timestamp
        values = {
            'latency_avg_ms': snapshot.latency_avg_ms,
            'latency_p95_ms': snapshot.latency_p95_ms,
            'api_error_rate': snapshot.api_error_rate,
            'slippage_mean_bps': snapshot.slippage_mean_bps,
            'spread_drift_bps': snapshot.spread_drift_bps,
            'intraday_drawdown_ratio': snapshot.intraday_drawdown_ratio,
            'gpu_memory_ratio': snapshot.gpu_memory_ratio,
            'redis_ok': 1.0 if snapshot.redis_ok else 0.0,
            'postgresql_ok': 1.0 if snapshot.postgresql_ok else 0.0,
        }

        for metric, value in values.items():
            series = self._series[metric]
            series.append((now, value))

        anomalies = self._detect_anomalies(now)
        actions = await self._notify_quant_kernel(quant_kernel, anomalies)
        self._publish(snapshot, anomalies)
        return HealthEvaluation(snapshot=snapshot, anomalies=anomalies, actions=actions)

    async def _check_redis(self, redis_client: Any) -> tuple[bool, float]:
        start = time.perf_counter()
        try:
            pong = await redis_client.ping()
            return bool(pong), (time.perf_counter() - start) * 1000
        except Exception:
            return False, 0.0

    async def _check_postgresql(self, database: Any) -> tuple[bool, float]:
        start = time.perf_counter()
        try:
            async with database.engine.begin() as conn:
                await conn.exec_driver_sql('SELECT 1')
            return True, (time.perf_counter() - start) * 1000
        except Exception:
            return False, 0.0

    def _detect_anomalies(self, now: float) -> list[HealthAnomaly]:
        anomalies: list[HealthAnomaly] = []
        for rule in self.rules:
            series = self._series[rule.metric]
            while series and (now - series[0][0]) > rule.window_seconds:
                series.popleft()
            if not series:
                continue

            window_values = [item[1] for item in series]
            aggregate = statistics.fmean(window_values)
            crit = self._breach(aggregate, rule.critical_threshold, rule.direction)
            warn = self._breach(aggregate, rule.warning_threshold, rule.direction)
            if not (crit or warn):
                continue
            severity: Severity = 'critical' if crit else 'warning'
            threshold = rule.critical_threshold if crit else rule.warning_threshold
            detail = (
                f"{rule.metric}={aggregate:.6f} breach({severity}) threshold={threshold:.6f} "
                f"window={int(rule.window_seconds)}s"
            )
            anomalies.append(
                HealthAnomaly(
                    metric=rule.metric,
                    severity=severity,
                    value=aggregate,
                    threshold=threshold,
                    window_seconds=rule.window_seconds,
                    detail=detail,
                )
            )
        return anomalies

    async def _notify_quant_kernel(self, quant_kernel: Any | None, anomalies: Sequence[HealthAnomaly]) -> list[str]:
        if quant_kernel is None or not anomalies:
            return []

        actions: list[str] = []
        if await _call_if_present(quant_kernel, ('activate_conservative_mode', 'enable_conservative_mode'), reason='system_health_anomaly'):
            actions.append('activate_conservative_mode')

        if await _call_if_present(quant_kernel, ('trim_capital', 'reduce_capital', 'recortar_capital'), factor=0.5):
            actions.append('trim_capital')

        has_critical = any(item.severity == 'critical' for item in anomalies)
        critical_drawdown = any(item.metric == 'intraday_drawdown_ratio' and item.severity == 'critical' for item in anomalies)
        infra_down = any(item.metric in {'redis_ok', 'postgresql_ok'} and item.severity == 'critical' for item in anomalies)
        if has_critical and (critical_drawdown or infra_down):
            if await _call_if_present(quant_kernel, ('emergency_shutdown',), reason='critical_system_health_anomaly'):
                actions.append('emergency_shutdown')
        return actions

    def _publish(self, snapshot: HealthSnapshot, anomalies: Sequence[HealthAnomaly]) -> None:
        if self.metrics is not None and hasattr(self.metrics, 'set_system_health_snapshot'):
            self.metrics.set_system_health_snapshot(snapshot)
        if self.metrics is not None and hasattr(self.metrics, 'observe_health_anomaly'):
            for anomaly in anomalies:
                self.metrics.observe_health_anomaly(anomaly.metric, anomaly.severity)

        payload = {
            'snapshot': asdict(snapshot),
            'anomalies': [asdict(a) for a in anomalies],
            'anomaly_count': len(anomalies),
        }
        bound = logger.bind(component='system_health', **payload)
        if any(item.severity == 'critical' for item in anomalies):
            bound.critical('System health anomaly detected')
        elif anomalies:
            bound.warning('System health anomaly detected')
        else:
            bound.info('System health nominal')

    @staticmethod
    def _breach(value: float, threshold: float, direction: Literal['gt', 'lt']) -> bool:
        if direction == 'lt':
            return value < threshold
        return value > threshold


async def _call_if_present(target: Any, names: Sequence[str], **kwargs: Any) -> bool:
    for name in names:
        member = getattr(target, name, None)
        if not callable(member):
            continue
        result = member(**kwargs)
        if inspect.isawaitable(result):
            await result
        return True
    return False


def _mean_or_zero(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _p95_or_zero(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    idx = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95))))
    return ordered[idx]


def _intraday_drawdown(equity_curve: Sequence[float]) -> float:
    peak = 0.0
    drawdown = 0.0
    for value in equity_curve:
        peak = max(peak, float(value))
        if peak <= 0:
            continue
        drawdown = max(drawdown, (peak - float(value)) / peak)
    return drawdown
