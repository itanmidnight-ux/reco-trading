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
    _http_server: object | None = None

    def start(self) -> None:
        self._http_server = start_http_server(self.port, addr=self.addr)

    def stop(self) -> None:
        if self._http_server is None:
            return

        if isinstance(self._http_server, tuple) and self._http_server:
            server = self._http_server[0]
            if hasattr(server, 'shutdown'):
                server.shutdown()
            if hasattr(server, 'server_close'):
                server.server_close()

        self._http_server = None


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

        # Métricas solicitadas para operación de trading
        self.worker_health = Gauge(
            'reco_worker_health',
            'Salud de worker (1=healthy, 0=degraded/down).',
            labelnames=('state', *_DEFAULT_LABELS),
            registry=self.registry,
        )
        self.fill_ratio = Gauge(
            'reco_fill_ratio',
            'Fill ratio de órdenes ejecutadas en rango [0, 1].',
            labelnames=('venue', *_DEFAULT_LABELS),
            registry=self.registry,
        )
        self.drawdown_ratio = Gauge(
            'reco_drawdown_ratio',
            'Drawdown actual en rango [0, 1].',
            labelnames=('scope', *_DEFAULT_LABELS),
            registry=self.registry,
        )
        # NOTE:
        # prometheus_client.Counter normaliza los nombres terminados en `_total`
        # y expone la familia sin el sufijo, lo cual rompe la convención utilizada
        # por el resto del sistema (dashboards/tests esperan `*_total` literal).
        # Usamos Gauge monotónico con `inc()` para mantener el nombre exacto.
        self.error_total = Gauge(
            'reco_errors_total',
            'Total de errores por componente y tipo.',
            labelnames=('component', 'error_type', *_DEFAULT_LABELS),
            registry=self.registry,
        )
        self.request_total = Gauge(
            'reco_requests_total',
            'Total de requests/eventos por componente.',
            labelnames=('component', *_DEFAULT_LABELS),
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

        self.system_latency_avg_ms = Gauge(
            'reco_system_latency_avg_ms',
            'Latencia promedio del sistema en milisegundos.',
            labelnames=_DEFAULT_LABELS,
            registry=self.registry,
        )
        self.system_latency_p95_ms = Gauge(
            'reco_system_latency_p95_ms',
            'Latencia p95 del sistema en milisegundos.',
            labelnames=_DEFAULT_LABELS,
            registry=self.registry,
        )
        self.system_api_error_rate = Gauge(
            'reco_system_api_error_rate',
            'Tasa de error de APIs en rango [0, 1].',
            labelnames=_DEFAULT_LABELS,
            registry=self.registry,
        )
        self.system_slippage_mean_bps = Gauge(
            'reco_system_slippage_mean_bps',
            'Slippage medio en bps.',
            labelnames=_DEFAULT_LABELS,
            registry=self.registry,
        )
        self.system_spread_drift_bps = Gauge(
            'reco_system_spread_drift_bps',
            'Deriva media del spread en bps.',
            labelnames=_DEFAULT_LABELS,
            registry=self.registry,
        )
        self.system_intraday_drawdown_ratio = Gauge(
            'reco_system_intraday_drawdown_ratio',
            'Drawdown intradía en rango [0, 1].',
            labelnames=_DEFAULT_LABELS,
            registry=self.registry,
        )
        self.system_redis_up = Gauge(
            'reco_system_redis_up',
            'Estado de Redis (1=up, 0=down).',
            labelnames=_DEFAULT_LABELS,
            registry=self.registry,
        )
        self.system_postgresql_up = Gauge(
            'reco_system_postgresql_up',
            'Estado de PostgreSQL (1=up, 0=down).',
            labelnames=_DEFAULT_LABELS,
            registry=self.registry,
        )
        self.system_gpu_memory_ratio = Gauge(
            'reco_system_gpu_memory_ratio',
            'Uso de memoria de GPU en rango [0, 1].',
            labelnames=_DEFAULT_LABELS,
            registry=self.registry,
        )
        self.system_health_anomaly_total = Counter(
            'reco_system_health_anomalies_total',
            'Total de anomalías de salud del sistema por métrica y severidad.',
            labelnames=('metric', 'severity', *_DEFAULT_LABELS),
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

    def set_fill_ratio(self, venue: str, value: float, **labels: str) -> None:
        self.fill_ratio.labels(**self._labels(venue=venue, **labels)).set(min(max(value, 0.0), 1.0))

    def set_drawdown(self, scope: str, value: float, **labels: str) -> None:
        self.drawdown_ratio.labels(**self._labels(scope=scope, **labels)).set(min(max(value, 0.0), 1.0))

    def observe_request(self, component: str, **labels: str) -> None:
        self.request_total.labels(**self._labels(component=component, **labels)).inc()

    def observe_error(self, component: str, error_type: str, **labels: str) -> None:
        self.error_total.labels(**self._labels(component=component, error_type=error_type, **labels)).inc()
        self.observe_request(component, **labels)

    def observe_fill_quality(self, fill_quality_ratio: float, slippage_bps: float, **labels: str) -> None:
        sample = self._labels(**labels)
        self.fill_quality_ratio.labels(**sample).observe(min(max(fill_quality_ratio, 0.0), 1.0))
        self.slippage_bps.labels(**sample).observe(slippage_bps)

    def set_system_health_snapshot(self, snapshot: object, **labels: str) -> None:
        sample = self._labels(**labels)
        self.system_latency_avg_ms.labels(**sample).set(float(getattr(snapshot, 'latency_avg_ms', 0.0)))
        self.system_latency_p95_ms.labels(**sample).set(float(getattr(snapshot, 'latency_p95_ms', 0.0)))
        self.system_api_error_rate.labels(**sample).set(min(max(float(getattr(snapshot, 'api_error_rate', 0.0)), 0.0), 1.0))
        self.system_slippage_mean_bps.labels(**sample).set(float(getattr(snapshot, 'slippage_mean_bps', 0.0)))
        self.system_spread_drift_bps.labels(**sample).set(float(getattr(snapshot, 'spread_drift_bps', 0.0)))
        self.system_intraday_drawdown_ratio.labels(**sample).set(
            min(max(float(getattr(snapshot, 'intraday_drawdown_ratio', 0.0)), 0.0), 1.0)
        )
        self.system_gpu_memory_ratio.labels(**sample).set(min(max(float(getattr(snapshot, 'gpu_memory_ratio', 0.0)), 0.0), 1.0))
        self.system_redis_up.labels(**sample).set(1.0 if bool(getattr(snapshot, 'redis_ok', False)) else 0.0)
        self.system_postgresql_up.labels(**sample).set(1.0 if bool(getattr(snapshot, 'postgresql_ok', False)) else 0.0)

    def observe_health_anomaly(self, metric: str, severity: str, **labels: str) -> None:
        self.system_health_anomaly_total.labels(**self._labels(metric=metric, severity=severity, **labels)).inc()

    def _labels(self, **custom_labels: str) -> dict[str, str]:
        defaults = {label: 'unknown' for label in _DEFAULT_LABELS}
        defaults.update({k: str(v) for k, v in custom_labels.items() if v is not None})
        return defaults
