import asyncio

from reco_trading.monitoring.metrics import TradingMetrics
from reco_trading.monitoring.system_health import HealthSnapshot, SystemHealthMonitor


class _DummyRedis:
    async def ping(self):
        return True


class _DummyConn:
    async def exec_driver_sql(self, _sql: str):
        return 1


class _DummyBegin:
    async def __aenter__(self):
        return _DummyConn()

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _DummyEngine:
    def begin(self):
        return _DummyBegin()


class _DummyDatabase:
    engine = _DummyEngine()


class _Kernel:
    def __init__(self):
        self.actions: list[str] = []

    async def activate_conservative_mode(self, **_kwargs):
        self.actions.append('activate_conservative_mode')

    async def trim_capital(self, **_kwargs):
        self.actions.append('trim_capital')

    async def emergency_shutdown(self, **_kwargs):
        self.actions.append('emergency_shutdown')


def test_collect_snapshot_computes_requested_indicators():
    monitor = SystemHealthMonitor()

    snapshot = asyncio.run(
        monitor.collect_snapshot(
            latencies_ms=[10.0, 30.0, 50.0, 70.0, 120.0],
            api_errors=2,
            api_requests=100,
            slippage_samples_bps=[1.0, 2.0, 3.0],
            spread_drift_samples_bps=[2.0, 4.0, 6.0],
            intraday_equity_curve=[100.0, 110.0, 105.0, 90.0],
            gpu_memory_used_bytes=9,
            gpu_memory_total_bytes=10,
            redis_client=_DummyRedis(),
            database=_DummyDatabase(),
        )
    )

    assert snapshot.latency_avg_ms > 0
    assert snapshot.latency_p95_ms >= 70.0
    assert snapshot.api_error_rate == 0.02
    assert snapshot.slippage_mean_bps == 2.0
    assert snapshot.spread_drift_bps == 4.0
    assert snapshot.intraday_drawdown_ratio > 0
    assert snapshot.gpu_memory_ratio == 0.9
    assert snapshot.redis_ok is True
    assert snapshot.postgresql_ok is True


def test_evaluate_triggers_quant_kernel_actions_for_critical_anomalies():
    monitor = SystemHealthMonitor()
    kernel = _Kernel()
    snapshot = HealthSnapshot(
        timestamp=1.0,
        latency_avg_ms=300.0,
        latency_p95_ms=350.0,
        api_error_rate=0.10,
        slippage_mean_bps=15.0,
        spread_drift_bps=20.0,
        intraday_drawdown_ratio=0.12,
        gpu_memory_ratio=0.99,
        redis_ok=False,
        postgresql_ok=False,
    )

    evaluation = asyncio.run(monitor.evaluate(snapshot, quant_kernel=kernel))

    assert len(evaluation.anomalies) >= 1
    severities = {item.severity for item in evaluation.anomalies}
    assert 'critical' in severities
    assert 'activate_conservative_mode' in evaluation.actions
    assert 'trim_capital' in evaluation.actions
    assert 'emergency_shutdown' in evaluation.actions


def test_publish_updates_metrics():
    metrics = TradingMetrics()
    monitor = SystemHealthMonitor(metrics=metrics)
    snapshot = HealthSnapshot(
        timestamp=2.0,
        latency_avg_ms=40.0,
        latency_p95_ms=90.0,
        api_error_rate=0.01,
        slippage_mean_bps=1.2,
        spread_drift_bps=1.5,
        intraday_drawdown_ratio=0.02,
        gpu_memory_ratio=0.7,
        redis_ok=True,
        postgresql_ok=True,
    )

    evaluation = asyncio.run(monitor.evaluate(snapshot))

    assert evaluation.snapshot.latency_avg_ms == 40.0
    assert evaluation.actions == []
