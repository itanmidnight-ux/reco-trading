from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from statistics import quantiles
from threading import RLock, Thread
from typing import Any
from http.server import BaseHTTPRequestHandler, HTTPServer


@dataclass
class RuntimeObservability:
    """Lightweight metrics collector with Prometheus-compatible text output."""

    latency_samples_ms: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    component_errors: Counter[str] = field(default_factory=Counter)
    reconnections: int = 0
    circuit_breaker_trips: int = 0
    stale_market_data_loops: int = 0
    total_loops: int = 0
    stage_latency_samples_ms: dict[str, deque[float]] = field(default_factory=dict)
    db_healthy: bool = False
    exchange_healthy: bool = False
    _lock: RLock = field(default_factory=RLock)

    def record_api_latency(self, latency_ms: float) -> None:
        with self._lock:
            self.latency_samples_ms.append(max(0.0, float(latency_ms)))

    def record_stage_latency(self, stage: str, latency_ms: float) -> None:
        normalized = str(stage or "unknown").strip().lower()
        if not normalized:
            normalized = "unknown"
        with self._lock:
            bucket = self.stage_latency_samples_ms.setdefault(normalized, deque(maxlen=200))
            bucket.append(max(0.0, float(latency_ms)))

    def record_error(self, component: str) -> None:
        with self._lock:
            self.component_errors[str(component)] += 1

    def record_reconnection(self) -> None:
        with self._lock:
            self.reconnections += 1

    def record_circuit_breaker_trip(self) -> None:
        with self._lock:
            self.circuit_breaker_trips += 1

    def record_loop(self, stale_market_data: bool) -> None:
        with self._lock:
            self.total_loops += 1
            if stale_market_data:
                self.stale_market_data_loops += 1

    def update_health(self, *, db_healthy: bool | None = None, exchange_healthy: bool | None = None) -> None:
        with self._lock:
            if db_healthy is not None:
                self.db_healthy = bool(db_healthy)
            if exchange_healthy is not None:
                self.exchange_healthy = bool(exchange_healthy)

    def latency_p95_ms(self) -> float:
        with self._lock:
            if len(self.latency_samples_ms) < 2:
                return float(self.latency_samples_ms[0]) if self.latency_samples_ms else 0.0
            bins = quantiles(list(self.latency_samples_ms), n=100, method="inclusive")
            return float(bins[94])

    def stale_ratio(self) -> float:
        with self._lock:
            if self.total_loops <= 0:
                return 0.0
            return float(self.stale_market_data_loops / self.total_loops)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            stage_latency_last_ms = {
                stage: float(samples[-1]) for stage, samples in self.stage_latency_samples_ms.items() if samples
            }
            stage_latency_p95_ms = {
                stage: _p95(samples) for stage, samples in self.stage_latency_samples_ms.items() if samples
            }
            return {
                "api_latency_p95_ms": self.latency_p95_ms(),
                "api_latency_last_ms": float(self.latency_samples_ms[-1]) if self.latency_samples_ms else 0.0,
                "reconnections": int(self.reconnections),
                "circuit_breaker_trips": int(self.circuit_breaker_trips),
                "stale_market_data_ratio": self.stale_ratio(),
                "db_healthy": int(self.db_healthy),
                "exchange_healthy": int(self.exchange_healthy),
                "component_errors": dict(self.component_errors),
                "stage_latency_last_ms": stage_latency_last_ms,
                "stage_latency_p95_ms": stage_latency_p95_ms,
            }

    def to_prometheus_text(self) -> str:
        snap = self.snapshot()
        lines = [
            "# TYPE reco_api_latency_p95_ms gauge",
            f"reco_api_latency_p95_ms {snap['api_latency_p95_ms']:.6f}",
            "# TYPE reco_stale_market_data_ratio gauge",
            f"reco_stale_market_data_ratio {snap['stale_market_data_ratio']:.6f}",
            "# TYPE reco_reconnections_total counter",
            f"reco_reconnections_total {snap['reconnections']}",
            "# TYPE reco_circuit_breaker_trips_total counter",
            f"reco_circuit_breaker_trips_total {snap['circuit_breaker_trips']}",
            "# TYPE reco_db_healthy gauge",
            f"reco_db_healthy {snap['db_healthy']}",
            "# TYPE reco_exchange_healthy gauge",
            f"reco_exchange_healthy {snap['exchange_healthy']}",
        ]
        for component, total in sorted(snap["component_errors"].items()):
            lines.append(f'reco_component_errors_total{{component="{component}"}} {int(total)}')
        for stage, latency in sorted(snap.get("stage_latency_p95_ms", {}).items()):
            lines.append(f'reco_stage_latency_p95_ms{{stage="{stage}"}} {float(latency):.6f}')
        return "\n".join(lines) + "\n"


def _p95(samples: deque[float]) -> float:
    if len(samples) < 2:
        return float(samples[0]) if samples else 0.0
    bins = quantiles(list(samples), n=100, method="inclusive")
    return float(bins[94])


class _ObservabilityHandler(BaseHTTPRequestHandler):
    registry: RuntimeObservability | None = None

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        payload = (self.registry.to_prometheus_text() if self.registry else "").encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def start_metrics_server(registry: RuntimeObservability, host: str, port: int) -> Thread:
    _ObservabilityHandler.registry = registry
    server = HTTPServer((host, int(port)), _ObservabilityHandler)
    thread = Thread(target=server.serve_forever, daemon=True, name="reco-observability")
    thread.start()
    return thread
