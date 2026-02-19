from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StructureSnapshot:
    trend: str
    hh_hl: bool
    lh_ll: bool
    momentum: float


class StructureEngine:
    def evaluate(self, close: list[float]) -> StructureSnapshot:
        if len(close) < 6:
            return StructureSnapshot('RANGE', False, False, 0.0)
        hh_hl = close[-1] > close[-2] > close[-3]
        lh_ll = close[-1] < close[-2] < close[-3]
        momentum = (close[-1] - close[-5]) / max(1e-9, close[-5])
        trend = 'BULL' if hh_hl else 'BEAR' if lh_ll else 'RANGE'
        return StructureSnapshot(trend, hh_hl, lh_ll, momentum)
