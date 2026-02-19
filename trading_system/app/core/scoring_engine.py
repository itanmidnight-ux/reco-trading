from __future__ import annotations

from dataclasses import dataclass

from trading_system.app.core.indicator_engine import IndicatorSnapshot
from trading_system.app.core.pattern_engine import PatternSnapshot
from trading_system.app.core.structure_engine import StructureSnapshot


@dataclass
class DeterministicScore:
    score: float
    breakdown: dict[str, float]


class ScoringEngine:
    def __init__(self, threshold: float = 0.62) -> None:
        self.threshold = threshold

    def score(self, ind: IndicatorSnapshot, pat: PatternSnapshot, struct: StructureSnapshot, relative_volume: float) -> DeterministicScore:
        points = {
            'trend_alignment': 0.2 if ind.ema20_gt_50 and ind.ema50_gt_200_proxy else 0.0,
            'rsi_zone': 0.15 if 45 <= ind.rsi <= 65 else 0.05 if 35 <= ind.rsi <= 75 else -0.05,
            'macd': 0.1 if ind.macd_hist > 0 else -0.1,
            'adx': 0.1 if ind.adx_proxy > 25 else 0.0,
            'pattern_breakout': 0.15 if pat.breakout else 0.0,
            'pattern_quality': 0.08 if (pat.engulfing or pat.pin_bar) else -0.03 if pat.doji else 0.0,
            'structure': 0.12 if struct.hh_hl else -0.12 if struct.lh_ll else 0.0,
            'momentum': max(-0.1, min(0.1, struct.momentum * 10)),
            'relative_volume': 0.1 if relative_volume > 1.2 else -0.05,
        }
        raw = sum(points.values())
        normalized = max(0.0, min(1.0, (raw + 0.4) / 1.2))
        return DeterministicScore(score=normalized, breakdown=points)
