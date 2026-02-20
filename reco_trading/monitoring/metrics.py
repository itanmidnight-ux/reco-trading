from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server

_DEFAULT_LABELS = ('exchange', 'strategy', 'worker_id', 'model_name')


@dataclass(slots=True)
class MetricsExporter:
    """Exportador de métricas para observabilidad en Prometheus."""

    port: int = 8001
    addr: str = '0.0.0.0'

    def start(self) -> None:
        start_http_server(self.port, addr=self.addr)


class TradingMetrics:
    def __init__(self, *, registry: CollectorRegistry | None = None, latency_buckets: Iterable[float] | None = None) -> None:
        self.registry = registry or CollectorRegistry(auto_describe=True)
        buckets = tuple(latency_buckets or (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0))

        self.stage_latency_seconds = Histogram(
            'reco_stage_latency_seconds',
            'Latencia por etapa para cálculo de p50/p95/p99 vía histogram_quantile.',
            labelnames=('stage', *_DEFAULT_LABELS),
            buckets=buckets,
            registry=self.registry,
        )
        self.stage_total = Counter(
            'reco_stage_events_total',
            'Cantidad de eventos procesados por etapa.',
            labelnames=('stage', *_DEFAULT_LABELS),
            registry=self.registry,
        )
        self.queue_occupancy = Gauge(
            'reco_queue_occupancy',
            'Ocupación actual de colas por etapa/worker.',
            labelnames=('queue_name', *_DEFAULT_LABELS),
            registry=self.registry,
        )
        self.gpu_utilization_ratio = Gauge(
            'reco_gpu_utilization_ratio',
            'Utilización de GPU en rango [0, 1].',
            labelnames=('gpu_id', *_DEFAULT_LABELS),
            registry=self.registry,
        )
        self.gpu_memory_bytes = Gauge(
            'reco_gpu_memory_bytes',
            'Memoria usada de GPU en bytes.',
            labelnames=('gpu_id', *_DEFAULT_LABELS),
            registry=self.registry,
        )
        self.worker_health = Gauge(
            'reco_worker_health',
            'Health de worker (1=healthy, 0=degraded/down).',
            labelnames=('state', *_DEFAULT_LABELS),
            registry=self.registry,
        )
        self.fill_quality_ratio = Histogram(
            'reco_fill_quality_ratio',
            'Fill quality en ratio [0, 1] (1 es mejor).',
            labelnames=_DEFAULT_LABELS,
            buckets=(0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0),
            registry=self.registry,
        )
        self.slippage_bps = Histogram(
            'reco_slippage_bps',
            'Slippage en basis points.',
            labelnames=_DEFAULT_LABELS,
            buckets=(-100, -50, -25, -10, -5, -1, 0, 1, 5, 10, 25, 50, 100),
            registry=self.registry,
        )

    def observe_stage_latency(self, stage: str, latency_seconds: float, **labels: str) -> None:
        sample = self._labels(stage=stage, **labels)
        self.stage_latency_seconds.labels(**sample).observe(max(latency_seconds, 0.0))
        self.stage_total.labels(**sample).inc()

    def set_queue_occupancy(self, queue_name: str, occupancy: int | float, **labels: str) -> None:
        self.queue_occupancy.labels(**self._labels(queue_name=queue_name, **labels)).set(float(max(occupancy, 0)))

    def set_gpu_stats(self, gpu_id: str, utilization_ratio: float, memory_bytes: int | float, **labels: str) -> None:
        sample = self._labels(gpu_id=gpu_id, **labels)
        self.gpu_utilization_ratio.labels(**sample).set(min(max(utilization_ratio, 0.0), 1.0))
        self.gpu_memory_bytes.labels(**sample).set(float(max(memory_bytes, 0)))

    def set_worker_health(self, state: str, healthy: bool, **labels: str) -> None:
        self.worker_health.labels(**self._labels(state=state, **labels)).set(1.0 if healthy else 0.0)

    def observe_fill_quality(self, fill_quality_ratio: float, slippage_bps: float, **labels: str) -> None:
        sample = self._labels(**labels)
        self.fill_quality_ratio.labels(**sample).observe(min(max(fill_quality_ratio, 0.0), 1.0))
        self.slippage_bps.labels(**sample).observe(slippage_bps)

    def _labels(self, **custom_labels: str) -> dict[str, str]:
        defaults = {label: 'unknown' for label in _DEFAULT_LABELS}
        defaults.update({k: str(v) for k, v in custom_labels.items() if v is not None})
        return defaults
