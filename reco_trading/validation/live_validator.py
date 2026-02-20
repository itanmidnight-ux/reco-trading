from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any, Protocol


class _RedisLike(Protocol):
    async def xadd(self, stream: str, fields: dict[str, str], maxlen: int | None = None, approximate: bool = True) -> Any: ...


class _DbLike(Protocol):
    async def persist_validation_event(self, payload: dict[str, Any]) -> None: ...


@dataclass(slots=True)
class ValidationThresholds:
    drift_warn: float = 0.30
    drift_critical: float = 0.60
    decay_warn: float = 0.15
    decay_critical: float = 0.30
    shadow_loss_warn: float = -0.01
    shadow_loss_critical: float = -0.03


@dataclass(slots=True)
class ValidationContract:
    status: str
    severity: str
    recommendations: list[str]
    metrics: dict[str, float]


@dataclass(slots=True)
class ValidationSnapshot:
    timestamp: float
    contract: ValidationContract
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LiveValidationInput:
    ts: float
    features: dict[str, float]
    signal: float
    confidence: float
    realized_return: float
    expected_return: float
    shadow_return: float


class LiveValidator:
    """Validador live con walk-forward rolling, shadow portfolio y detección de degradación."""

    def __init__(
        self,
        *,
        quant_kernel: Any,
        db: _DbLike | None = None,
        redis_client: _RedisLike | None = None,
        redis_stream: str = 'reco_trading:validation_history',
        rolling_window: int = 120,
        thresholds: ValidationThresholds | None = None,
    ) -> None:
        self.quant_kernel = quant_kernel
        self.db = db
        self.redis_client = redis_client
        self.redis_stream = redis_stream
        self.rolling_window = max(rolling_window, 30)
        self.thresholds = thresholds or ValidationThresholds()
        self._history: list[LiveValidationInput] = []

    async def validate_tick(self, point: LiveValidationInput) -> ValidationSnapshot:
        self._history.append(point)
        self._history = self._history[-self.rolling_window :]

        metrics = self._compute_metrics()
        contract = self._build_contract(metrics)
        snapshot = ValidationSnapshot(timestamp=point.ts, contract=contract, context={'window_size': len(self._history)})

        self.quant_kernel.notify_validation_event(contract.severity, contract.status, asdict(snapshot))
        self.quant_kernel.reduce_capital_for_validation(contract.severity)
        self.quant_kernel.activate_conservative_mode(contract.severity)

        await self._persist(snapshot)
        return snapshot

    def _compute_metrics(self) -> dict[str, float]:
        if len(self._history) < 5:
            return {
                'walk_forward_score': 1.0,
                'shadow_pnl': 0.0,
                'feature_drift': 0.0,
                'signal_drift': 0.0,
                'ic_decay': 0.0,
                'alpha_decay': 0.0,
                'confidence_decay': 0.0,
            }

        half = max(2, len(self._history) // 2)
        train = self._history[:half]
        test = self._history[half:]

        walk_forward_score = mean([1.0 - abs(x.expected_return - x.realized_return) for x in test])
        shadow_pnl = sum(x.shadow_return for x in self._history)

        feature_drift = self._distribution_shift(
            [mean(list(x.features.values())) if x.features else 0.0 for x in train],
            [mean(list(x.features.values())) if x.features else 0.0 for x in test],
        )
        signal_drift = self._distribution_shift([x.signal for x in train], [x.signal for x in test])

        ic_train = self._correlation([x.signal for x in train], [x.realized_return for x in train])
        ic_test = self._correlation([x.signal for x in test], [x.realized_return for x in test])
        ic_decay = max(0.0, ic_train - ic_test)

        alpha_train = mean([x.expected_return for x in train])
        alpha_test = mean([x.realized_return for x in test])
        alpha_decay = max(0.0, alpha_train - alpha_test)

        conf_train = mean([x.confidence for x in train])
        conf_test = mean([x.confidence for x in test])
        confidence_decay = max(0.0, conf_train - conf_test)

        return {
            'walk_forward_score': float(walk_forward_score),
            'shadow_pnl': float(shadow_pnl),
            'feature_drift': float(feature_drift),
            'signal_drift': float(signal_drift),
            'ic_decay': float(ic_decay),
            'alpha_decay': float(alpha_decay),
            'confidence_decay': float(confidence_decay),
        }

    def _build_contract(self, metrics: dict[str, float]) -> ValidationContract:
        score = 0
        recommendations: list[str] = []

        drift_peak = max(metrics['feature_drift'], metrics['signal_drift'])
        if drift_peak >= self.thresholds.drift_critical:
            score += 2
            recommendations.append('Reentrenar features y recalibrar señales inmediatamente.')
        elif drift_peak >= self.thresholds.drift_warn:
            score += 1
            recommendations.append('Monitorear drift y aumentar frecuencia de retraining.')

        if metrics['shadow_pnl'] <= self.thresholds.shadow_loss_critical:
            score += 2
            recommendations.append('Pausar despliegues de nuevas estrategias y revisar fill assumptions del shadow portfolio.')
        elif metrics['shadow_pnl'] <= self.thresholds.shadow_loss_warn:
            score += 1
            recommendations.append('Reducir exposición hasta recuperar PnL en shadow portfolio.')

        decay_peak = max(metrics['ic_decay'], metrics['alpha_decay'], metrics['confidence_decay'])
        if decay_peak >= self.thresholds.decay_critical:
            score += 2
            recommendations.append('Activar rollback de modelos: decay crítico detectado.')
        elif decay_peak >= self.thresholds.decay_warn:
            score += 1
            recommendations.append('Ajustar hiperparámetros y elevar filtros de entrada por decay.')

        if score >= 4:
            severity = 'critico'
            status = 'crítico'
            recommendations.append('Operar en modo ultra conservador con mínimo capital.')
        elif score >= 2:
            severity = 'degradado'
            status = 'degradado'
            recommendations.append('Activar modo conservador y límites de riesgo reforzados.')
        else:
            severity = 'normal'
            status = 'normal'
            recommendations.append('Mantener régimen normal con vigilancia estándar.')

        return ValidationContract(status=status, severity=severity, recommendations=recommendations, metrics=metrics)

    async def _persist(self, snapshot: ValidationSnapshot) -> None:
        payload = asdict(snapshot)
        if self.db is not None:
            await self.db.persist_validation_event(payload)
        if self.redis_client is not None:
            await self.redis_client.xadd(self.redis_stream, {'payload': str(payload)}, maxlen=5000, approximate=True)

    @staticmethod
    def _distribution_shift(a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        return abs(mean(a) - mean(b)) / max(1e-9, abs(mean(a)) + abs(mean(b)) + 1e-9)

    @staticmethod
    def _correlation(x: list[float], y: list[float]) -> float:
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        mx, my = mean(x), mean(y)
        cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y, strict=False))
        sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
        sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
        denom = sx * sy
        if denom <= 1e-12:
            return 0.0
        return cov / denom
