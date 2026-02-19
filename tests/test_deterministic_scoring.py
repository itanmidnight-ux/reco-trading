from trading_system.app.core.indicator_engine import IndicatorSnapshot
from trading_system.app.core.pattern_engine import PatternSnapshot
from trading_system.app.core.scoring_engine import ScoringEngine
from trading_system.app.core.structure_engine import StructureSnapshot


def test_deterministic_scoring_bounds():
    engine = ScoringEngine()
    out = engine.score(
        IndicatorSnapshot(55, 0.2, True, True, 1.0, 30, 0.3),
        PatternSnapshot(True, False, False, True, False),
        StructureSnapshot('BULL', True, False, 0.01),
        relative_volume=1.4,
    )
    assert 0 <= out.score <= 1
    assert 'trend_alignment' in out.breakdown
