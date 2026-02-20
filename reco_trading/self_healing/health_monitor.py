from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from reco_trading.monitoring.alert_manager import AlertManager
from reco_trading.self_healing.anomaly_detection import AnomalyDetectionEngine
from reco_trading.self_healing.capital_protection import CapitalProtectionEngine
from reco_trading.self_healing.recovery_engine import RecoveryAction, RecoveryEngine


@dataclass(slots=True)
class HealthCheckResult:
    check: str
    status: str
    severity: str
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class IncidentRecord:
    incident_id: str
    severity: str
    title: str
    detail: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class IncidentAuditLog:
    def __init__(self, path: str = 'logs/self_healing_incidents.jsonl') -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: IncidentRecord) -> None:
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + '\n')


class HealthMonitor:
    def __init__(
        self,
        *,
        alert_manager: AlertManager,
        anomaly_detector: AnomalyDetectionEngine,
        recovery_engine: RecoveryEngine,
        capital_protection: CapitalProtectionEngine,
        audit_log: IncidentAuditLog | None = None,
    ) -> None:
        self.alert_manager = alert_manager
        self.anomaly_detector = anomaly_detector
        self.recovery_engine = recovery_engine
        self.capital_protection = capital_protection
        self.audit_log = audit_log or IncidentAuditLog()

    def check_exchange_latency(self, latency_by_exchange: dict[str, float], *, threshold_ms: float = 250.0) -> list[HealthCheckResult]:
        results: list[HealthCheckResult] = []
        for exchange, latency_ms in latency_by_exchange.items():
            status = 'ok' if latency_ms <= threshold_ms else 'degraded'
            severity = 'info' if status == 'ok' else 'warning'
            detail = f'latencia={latency_ms:.2f}ms'
            results.append(HealthCheckResult('exchange_latency', status, severity, detail, {'exchange': exchange, 'latency_ms': latency_ms}))
        return results

    def check_book_ticker_desync(self, snapshots: dict[str, dict[str, float]], *, threshold_bps: float = 12.0) -> list[HealthCheckResult]:
        results: list[HealthCheckResult] = []
        for exchange, item in snapshots.items():
            book_mid = float(item.get('book_mid', 0.0))
            ticker = float(item.get('ticker', 0.0))
            drift_bps = 0.0 if book_mid <= 0 else abs(book_mid - ticker) / book_mid * 10_000
            status = 'ok' if drift_bps <= threshold_bps else 'degraded'
            severity = 'info' if status == 'ok' else 'warning'
            detail = f'drift={drift_bps:.2f}bps'
            results.append(
                HealthCheckResult('book_ticker_desync', status, severity, detail, {'exchange': exchange, 'drift_bps': drift_bps})
            )
        return results

    def check_api_error_rate(self, error_rate_by_exchange: dict[str, float], *, threshold: float = 0.03) -> list[HealthCheckResult]:
        return [
            HealthCheckResult(
                'api_error_rate',
                'ok' if rate <= threshold else 'degraded',
                'info' if rate <= threshold else 'warning',
                f'error_rate={rate:.2%}',
                {'exchange': exchange, 'error_rate': rate},
            )
            for exchange, rate in error_rate_by_exchange.items()
        ]

    async def check_redis_health(self, redis_client: Any) -> HealthCheckResult:
        start = time.perf_counter()
        try:
            pong = await redis_client.ping()
            elapsed_ms = (time.perf_counter() - start) * 1000
            status = 'ok' if pong else 'failed'
            return HealthCheckResult('redis_health', status, 'info' if pong else 'critical', f'ping={pong}', {'latency_ms': elapsed_ms})
        except Exception as err:
            return HealthCheckResult('redis_health', 'failed', 'critical', f'error={err}', {})

    async def check_postgresql_health(self, db: Any) -> HealthCheckResult:
        start = time.perf_counter()
        try:
            async with db.engine.begin() as conn:
                await conn.exec_driver_sql('SELECT 1')
            elapsed_ms = (time.perf_counter() - start) * 1000
            return HealthCheckResult('postgresql_health', 'ok', 'info', 'SELECT 1 OK', {'latency_ms': elapsed_ms})
        except Exception as err:
            return HealthCheckResult('postgresql_health', 'failed', 'critical', f'error={err}', {})

    def check_worker_cluster_health(self, workers: list[dict[str, Any]], *, stale_after_s: float = 15.0) -> HealthCheckResult:
        now = time.time()
        stale = [w['worker_id'] for w in workers if now - float(w.get('last_heartbeat_ts', 0.0)) > stale_after_s]
        overloaded = [w['worker_id'] for w in workers if float(w.get('load', 0.0)) >= 0.9]
        status = 'ok' if not stale and not overloaded else 'degraded'
        severity = 'info' if status == 'ok' else 'warning'
        detail = f'stale={len(stale)} overloaded={len(overloaded)}'
        return HealthCheckResult(
            'worker_cluster_health',
            status,
            severity,
            detail,
            {'stale_workers': stale, 'overloaded_workers': overloaded, 'cluster_size': len(workers)},
        )

    def check_gpu_telemetry(
        self,
        telemetry: dict[str, dict[str, float]],
        *,
        memory_threshold: float = 0.9,
        inference_error_threshold: float = 0.01,
    ) -> list[HealthCheckResult]:
        results: list[HealthCheckResult] = []
        for device_id, stats in telemetry.items():
            mem_ratio = float(stats.get('memory_ratio', 0.0))
            inf_error_rate = float(stats.get('inference_error_rate', 0.0))
            degraded = mem_ratio >= memory_threshold or inf_error_rate >= inference_error_threshold
            severity = 'warning' if degraded else 'info'
            results.append(
                HealthCheckResult(
                    'gpu_telemetry',
                    'degraded' if degraded else 'ok',
                    severity,
                    f'memory={mem_ratio:.2%} inference_error={inf_error_rate:.2%}',
                    {'device_id': device_id, 'memory_ratio': mem_ratio, 'inference_error_rate': inf_error_rate},
                )
            )
        return results

    def _raise_incident(self, result: HealthCheckResult) -> IncidentRecord:
        incident = IncidentRecord(
            incident_id=f"incident-{int(result.timestamp * 1000)}-{result.check}",
            severity=result.severity,
            title=result.check,
            detail=result.detail,
            payload=result.metrics,
            timestamp=result.timestamp,
        )
        self.alert_manager.emit(result.check, result.detail, severity='error' if result.severity == 'critical' else 'warning', payload=result.metrics)
        self.audit_log.append(incident)
        logger.bind(component='self_healing', incident_id=incident.incident_id, check=result.check).warning('Incidente registrado en auditoría')
        return incident

    def evaluate_anomalies(self, metric_series: dict[str, list[float]]) -> list[IncidentRecord]:
        incidents: list[IncidentRecord] = []
        for metric_name, values in metric_series.items():
            detections = self.anomaly_detector.detect(values)
            for detection in detections:
                if not detection.is_anomaly:
                    continue
                result = HealthCheckResult(
                    check=f'anomaly_{metric_name}_{detection.method}',
                    status='degraded',
                    severity='warning',
                    detail=f'Anomalía detectada: score={detection.score:.4f}',
                    metrics={'metric': metric_name, 'method': detection.method, 'score': detection.score, **detection.metadata},
                )
                incidents.append(self._raise_incident(result))
        return incidents

    def execute_recovery_policy(self, incidents: list[IncidentRecord]) -> list[RecoveryAction]:
        actions: list[RecoveryAction] = []
        for incident in incidents:
            title = incident.title
            if 'postgresql' in title:
                actions.append(self.recovery_engine.restart_module('postgres_writer'))
                actions.append(self.recovery_engine.rollback_to_stable_configuration())
            elif 'redis' in title or 'worker_cluster' in title:
                actions.append(self.recovery_engine.restart_module('cluster_coordinator'))
                actions.append(self.recovery_engine.switch_to_backup_exchange('primary', 'backup'))
            elif 'gpu' in title:
                actions.append(self.recovery_engine.disable_strategy_temporarily('directional'))
            elif 'api_error_rate' in title:
                actions.append(self.recovery_engine.activate_conservative_fallback())
        return actions

    async def run_cycle(
        self,
        *,
        latency_by_exchange: dict[str, float],
        desync_snapshots: dict[str, dict[str, float]],
        api_error_rate_by_exchange: dict[str, float],
        workers: list[dict[str, Any]],
        gpu_telemetry: dict[str, dict[str, float]],
        redis_client: Any,
        database: Any,
        anomaly_series: dict[str, list[float]] | None = None,
    ) -> dict[str, Any]:
        checks: list[HealthCheckResult] = []
        checks.extend(self.check_exchange_latency(latency_by_exchange))
        checks.extend(self.check_book_ticker_desync(desync_snapshots))
        checks.extend(self.check_api_error_rate(api_error_rate_by_exchange))
        checks.append(self.check_worker_cluster_health(workers))
        checks.extend(self.check_gpu_telemetry(gpu_telemetry))

        redis_result, pg_result = await asyncio.gather(
            self.check_redis_health(redis_client),
            self.check_postgresql_health(database),
        )
        checks.extend([redis_result, pg_result])

        incidents = [self._raise_incident(result) for result in checks if result.status != 'ok']
        if anomaly_series:
            incidents.extend(self.evaluate_anomalies(anomaly_series))

        if any(incident.severity == 'critical' for incident in incidents):
            severe_action = self.capital_protection.activate_severe_defensive_mode('critical_incident_detected')
            self.audit_log.append(
                IncidentRecord(
                    incident_id=f'defensive-{int(time.time() * 1000)}',
                    severity='critical',
                    title='capital_protection',
                    detail=severe_action.detail,
                    payload=asdict(severe_action.state),
                )
            )

        recovery_actions = self.execute_recovery_policy(incidents)
        return {
            'checks': [asdict(c) for c in checks],
            'incidents': [asdict(i) for i in incidents],
            'recovery_actions': [asdict(a) for a in recovery_actions],
            'defensive_state': asdict(self.capital_protection.snapshot()),
        }
