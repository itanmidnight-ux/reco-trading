from __future__ import annotations

from dataclasses import dataclass

from reco_trading.analysis.indicators import IndicatorSet


@dataclass(frozen=True, slots=True)
class MarketRegime:
    name: str
    volatility_extreme: bool
    tradable: bool


def detect_regime(indicators: IndicatorSet, max_spread_bps: float = 20.0) -> MarketRegime:
    high_vol = indicators.ewma_vol_annualized > 1.2
    trend = indicators.adx14 >= 25.0
    range_market = indicators.adx14 < 20.0
    low_activity = indicators.relative_volume_percentile < 0.2

    if indicators.spread_bps > max_spread_bps or low_activity:
        return MarketRegime(name='ILLIQUID', volatility_extreme=high_vol, tradable=False)
    if high_vol:
        return MarketRegime(name='HIGH_VOL', volatility_extreme=True, tradable=False)
    if trend:
        return MarketRegime(name='TREND', volatility_extreme=False, tradable=True)
    if range_market:
        return MarketRegime(name='RANGE', volatility_extreme=False, tradable=True)
    return MarketRegime(name='NEUTRAL', volatility_extreme=False, tradable=True)
