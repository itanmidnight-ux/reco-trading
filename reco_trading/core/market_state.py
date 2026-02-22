from __future__ import annotations

from dataclasses import dataclass

from reco_trading.analysis.indicators import IndicatorSet
from reco_trading.analysis.regimes import MarketRegime


@dataclass(frozen=True, slots=True)
class MarketState:
    price: float
    indicators: IndicatorSet
    regime: MarketRegime
    expected_edge: float
    friction_cost: float
    timestamp_ms: int
