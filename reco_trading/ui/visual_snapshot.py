from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class VisualSnapshot:
    capital: float
    balance: float
    pnl_total: float
    pnl_diario: float
    drawdown: float
    riesgo_activo: float
    exposicion: float
    trades: int
    win_rate: float
    expectancy: float
    sharpe_rolling: float
    regimen: str
    senal: str
    latencia_ms: float
    ultimo_precio: float
    estado_binance: str
    estado_sistema: str
    actividad: str
    motivo_bloqueo: str
    confianza: float = 0.0
    tiempo_en_posicion_s: float = 0.0
    cooldown_restante_s: float = 0.0
    score_momentum: float = 0.5
    score_reversion: float = 0.5
    score_regime: float = 0.5
    learning_remaining_seconds: float = 0.0


__all__ = ['VisualSnapshot']
