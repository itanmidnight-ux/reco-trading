from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class RiskSignal:
    blocked: bool
    reason: str
    source: str
    priority: int = 100


@dataclass(slots=True)
class RiskVerdictResult:
    blocked: bool
    reason: str
    source: str


class RiskVerdictEngine:
    """Consolida todas las rutas de veto en un único dictamen priorizado."""

    def evaluate(self, signals: Iterable[RiskSignal]) -> RiskVerdictResult:
        ordered = sorted(signals, key=lambda s: s.priority)
        for signal in ordered:
            if signal.blocked:
                return RiskVerdictResult(True, signal.reason, signal.source)
        return RiskVerdictResult(False, 'none', 'ok')
