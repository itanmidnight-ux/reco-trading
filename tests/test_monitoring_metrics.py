import pytest

prom = pytest.importorskip('prometheus_client')
CollectorRegistry = prom.CollectorRegistry

from reco_trading.monitoring.metrics import TradingMetrics


def test_trading_metrics_observes_lifecycle_and_labels():
    registry = CollectorRegistry(auto_describe=True)
    metrics = TradingMetrics(registry=registry)

    labels = {
        'exchange': 'binance',
        'strategy': 'mm',
        'worker_id': 'w1',
        'model_name': 'ofm',
    }
    metrics.observe_stage_latency('inference', 0.012, **labels)
    metrics.set_queue_occupancy('worker_q', 7, **labels)
    metrics.set_gpu_stats('gpu-0', 0.7, 1024, **labels)
    metrics.set_worker_health('healthy', True, **labels)
    metrics.observe_fill_quality(0.93, 2.1, **labels)

    families = {family.name for family in registry.collect()}
    assert 'reco_stage_latency_seconds' in families
    assert 'reco_queue_occupancy' in families
    assert 'reco_gpu_utilization_ratio' in families
    assert 'reco_fill_quality_ratio' in families
