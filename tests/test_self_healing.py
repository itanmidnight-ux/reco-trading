from __future__ import annotations

from pathlib import Path

import asyncio

from reco_trading.monitoring.alert_manager import AlertManager
from reco_trading.self_healing.anomaly_detection import AnomalyDetectionEngine
from reco_trading.self_healing.capital_protection import CapitalProtectionEngine
from reco_trading.self_healing.health_monitor import HealthMonitor, IncidentAuditLog
from reco_trading.self_healing.recovery_engine import RecoveryEngine


class _DummyRedis:
    async def ping(self) -> bool:
        return True


class _DummyConnection:
    async def __aenter__(self) -> _DummyConnection:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def exec_driver_sql(self, query: str) -> None:
        assert query == 'SELECT 1'


class _DummyEngine:
    def begin(self) -> _DummyConnection:
        return _DummyConnection()


class _DummyDatabase:
    engine = _DummyEngine()


def test_anomaly_detection_detects_large_spike() -> None:
    detector = AnomalyDetectionEngine(z_threshold=2.0)
    series = [100.0 + (i % 4) for i in range(140)] + [170.0]

    results = detector.detect(series)

    assert len(results) == 3
    assert any(result.is_anomaly for result in results)


def test_health_monitor_writes_auditable_incidents(tmp_path: Path) -> None:
    audit_log = IncidentAuditLog(path=str(tmp_path / 'incidents.jsonl'))
    monitor = HealthMonitor(
        alert_manager=AlertManager(),
        anomaly_detector=AnomalyDetectionEngine(),
        recovery_engine=RecoveryEngine(),
        capital_protection=CapitalProtectionEngine(),
        audit_log=audit_log,
    )

    output = asyncio.run(
        monitor.run_cycle(
            latency_by_exchange={'binance': 420.0},
            desync_snapshots={'binance': {'book_mid': 100.0, 'ticker': 90.0}},
            api_error_rate_by_exchange={'binance': 0.05},
            workers=[{'worker_id': 'w-1', 'last_heartbeat_ts': 0.0, 'load': 0.95}],
            gpu_telemetry={'gpu0': {'memory_ratio': 0.95, 'inference_error_rate': 0.02}},
            redis_client=_DummyRedis(),
            database=_DummyDatabase(),
            anomaly_series={'latency': [100.0 + (i % 3) for i in range(140)] + [400.0]},
        )
    )

    assert len(output['incidents']) >= 5
    assert (tmp_path / 'incidents.jsonl').exists()
    assert (tmp_path / 'incidents.jsonl').read_text(encoding='utf-8').count('\n') >= 5
