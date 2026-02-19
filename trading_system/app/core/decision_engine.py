from __future__ import annotations

from dataclasses import dataclass

from trading_system.app.core.scoring_engine import DeterministicScore


@dataclass
class DeterministicDecision:
    signal: str
    confidence: float
    reason: str


class DeterministicDecisionEngine:
    def decide(self, deterministic: DeterministicScore, min_structural_score: float = 0.55) -> DeterministicDecision:
        s = deterministic.score
        if s >= max(0.75, min_structural_score):
            signal = 'LONG'
        elif s <= 0.25:
            signal = 'SHORT'
        else:
            signal = 'HOLD'
        conf = abs(s - 0.5) * 2
        return DeterministicDecision(signal=signal, confidence=conf, reason=f'deterministic_score={s:.3f}')
