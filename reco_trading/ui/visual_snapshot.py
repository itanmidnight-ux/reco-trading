from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class VisualSnapshot:
    price: float
    equity: float
    pnl: float
    decision: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    regime: str = 'UNKNOWN'
    risk_state: str = 'OK'
    execution_state: str = 'IDLE'
    reason: str = ''
    system_state: str = 'BOOTING'
    daily_pnl: float = 0.0
    drawdown: float = 0.0
    expectancy: float = 0.0
    regime_expectancy: float = 0.0
    volatility: float = 0.0
    edge: float = 0.0
    edge_confidence_score: float = 0.0
    edge_t_stat: float = 0.0
    edge_bayesian_prob: float = 0.0
    edge_sprt_state: str = 'INCONCLUSIVE'
    risk_of_ruin_probability: float = 1.0
    regime_stability_score: float = 0.0
    model_diagnostics: dict[str, dict[str, float]] = field(default_factory=dict)
    critical_error: str = ''


__all__ = ['VisualSnapshot']
