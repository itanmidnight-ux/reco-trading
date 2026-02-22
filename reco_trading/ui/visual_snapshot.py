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


__all__ = ['VisualSnapshot']
