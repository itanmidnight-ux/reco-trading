from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass(slots=True)
class SLODefinition:
    name: str
    objective: float
    window: str
    description: str


class AlertManager:
    def emit(self, title: str, detail: str, *, severity: str = 'error', exchange: str | None = None, payload: dict[str, Any] | None = None) -> None:
        bound = logger.bind(component='alert_manager', title=title, severity=severity, exchange=exchange, payload=payload or {})
        log_fn = getattr(bound, severity, bound.error)
        log_fn(f'ALERT | {title} | {detail}')

    def evaluate_slo_alerts(
        self,
        *,
        error_rate: float,
        p95_latency_seconds: float,
        fill_ratio: float,
        drawdown_ratio: float,
        capital_protection_active: bool,
        exchange: str | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> None:
        payload = dict(extra_payload or {})
        payload.update(
            {
                'error_rate': float(error_rate),
                'p95_latency_seconds': float(p95_latency_seconds),
                'fill_ratio': float(fill_ratio),
                'drawdown_ratio': float(drawdown_ratio),
                'capital_protection_active': bool(capital_protection_active),
            }
        )

        # Error budget: SLO disponibilidad 99.5% => error_budget = 0.5%
        if error_rate > 0.005:
            self.emit(
                'error budget burn',
                f'Error rate {error_rate:.4%} supera presupuesto (0.50%).',
                severity='critical',
                exchange=exchange,
                payload=payload,
            )

        # Breach de latencia operativa
        if p95_latency_seconds > 0.050:
            self.emit(
                'latency breach',
                f'p95 de latencia {p95_latency_seconds:.3f}s excede umbral 0.050s.',
                severity='warning',
                exchange=exchange,
                payload=payload,
            )

        # Anomalía de ejecución por fill ratio bajo
        if fill_ratio < 0.92:
            self.emit(
                'execution anomaly',
                f'Fill ratio {fill_ratio:.2%} por debajo del SLO 92%.',
                severity='warning',
                exchange=exchange,
                payload=payload,
            )

        # Protección de capital por drawdown o kill switch
        if capital_protection_active or drawdown_ratio >= 0.12:
            self.emit(
                'capital protection active',
                'Se activó protección de capital por drawdown o kill-switch.',
                severity='critical',
                exchange=exchange,
                payload=payload,
            )


SLO_DEFINITIONS: tuple[SLODefinition, ...] = (
    SLODefinition(
        name='error_budget',
        objective=0.995,
        window='30d',
        description='Disponibilidad del pipeline >= 99.5% (error rate <= 0.5%).',
    ),
    SLODefinition(
        name='latency_p95',
        objective=0.050,
        window='1h rolling',
        description='Latencia p95 de etapas críticas <= 50ms.',
    ),
    SLODefinition(
        name='execution_fill_ratio',
        objective=0.92,
        window='1h rolling',
        description='Fill ratio agregado >= 92%.',
    ),
    SLODefinition(
        name='capital_drawdown_guard',
        objective=0.12,
        window='1d rolling',
        description='Drawdown diario debe mantenerse por debajo de 12%.',
    ),
)
