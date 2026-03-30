from __future__ import annotations

import json
import logging
import socket
import sys
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from statistics import quantiles
from threading import RLock, Thread
from typing import Any
from http.server import BaseHTTPRequestHandler, HTTPServer


class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter for machine parsing."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra"):
            log_obj.update(record.extra)

        return json.dumps(log_obj)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_file: str | None = None,
    log_rotation_max_bytes: int = 10 * 1024 * 1024,
    log_rotation_backup_count: int = 5,
    structured: bool = False,
) -> logging.Logger:
    """
    Configure logging with console and file handlers.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        log_rotation_max_bytes: Max size per log file
        log_rotation_backup_count: Number of backup files
        structured: Use JSON structured logging

    Returns:
        Configured root logger
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if structured:
        console_formatter = StructuredFormatter()
    else:
        console_formatter = ColoredFormatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    if log_file:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=log_rotation_max_bytes,
            backupCount=log_rotation_backup_count,
        )
        file_handler.setLevel(logging.DEBUG)

        if structured:
            file_formatter = StructuredFormatter()
        else:
            file_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    return root_logger


@dataclass
class RuntimeObservability:
    """Lightweight metrics collector with Prometheus-compatible text output."""

    latency_samples_ms: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    component_errors: Counter[str] = field(default_factory=Counter)
    reconnections: int = 0
    circuit_breaker_trips: int = 0
    stale_market_data_loops: int = 0
    total_loops: int = 0
    db_healthy: bool = False
    exchange_healthy: bool = False
    _lock: RLock = field(default_factory=RLock)

    def record_api_latency(self, latency_ms: float) -> None:
        with self._lock:
            self.latency_samples_ms.append(max(0.0, float(latency_ms)))

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
            return {
                "api_latency_p95_ms": self.latency_p95_ms(),
                "api_latency_last_ms": float(self.latency_samples_ms[-1]) if self.latency_samples_ms else 0.0,
                "reconnections": int(self.reconnections),
                "circuit_breaker_trips": int(self.circuit_breaker_trips),
                "stale_market_data_ratio": self.stale_ratio(),
                "db_healthy": int(self.db_healthy),
                "exchange_healthy": int(self.exchange_healthy),
                "component_errors": dict(self.component_errors),
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
        return "\n".join(lines) + "\n"


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


_metrics_server_instance: HTTPServer | None = None


def start_metrics_server(registry: RuntimeObservability, host: str, port: int) -> Thread:
    global _metrics_server_instance
    
    _ObservabilityHandler.registry = registry
    
    class ReusableHTTPServer(HTTPServer):
        def server_bind(self) -> None:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            super().server_bind()
    
    if _metrics_server_instance is not None:
        try:
            _metrics_server_instance.shutdown()
            _metrics_server_instance.server_close()
        except Exception:
            pass
        _metrics_server_instance = None
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            server = ReusableHTTPServer((host, int(port)), _ObservabilityHandler)
            _metrics_server_instance = server
            thread = Thread(target=server.serve_forever, daemon=True, name="reco-observability")
            thread.start()
            return thread
        except OSError as e:
            if e.errno == 98 and attempt < max_retries - 1:
                import time
                time.sleep(0.5 * (attempt + 1))
                continue
            raise
    
    raise RuntimeError(f"Failed to start metrics server on {host}:{port}")
