import pandas as pd

from reco_trading.strategy.market_intelligence import MarketRangePositionFilter, MarketRegime


def _range_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "low": [90.0 + (i % 3) for i in range(130)],
            "high": [110.0 - (i % 3) for i in range(130)],
        }
    )


def test_near_resistance_in_trending_allows_trade_without_hard_block() -> None:
    filt = MarketRangePositionFilter()

    result = filt.assess(
        df=_range_frame(),
        side="BUY",
        price=109.8,
        market_regime=MarketRegime.TRENDING,
        adx=30.0,
    )

    assert result.allow_trade is True
    assert result.range_multiplier == 0.30


def test_near_resistance_in_ranging_allows_trade_with_lower_multiplier() -> None:
    filt = MarketRangePositionFilter()

    result = filt.assess(
        df=_range_frame(),
        side="BUY",
        price=109.8,
        market_regime=MarketRegime.RANGING,
        adx=15.0,
    )

    assert result.allow_trade is True
    assert result.range_multiplier == 0.15
