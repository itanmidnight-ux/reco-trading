from __future__ import annotations

import pandas as pd

from reco_trading.strategy.market_intelligence import LiquidityZoneDetector, MarketIntelligence, VolatilityFilter


class _Settings:
    enable_market_intelligence = True
    volatility_filter_enabled = True
    liquidity_zone_filter_enabled = True
    market_regime_classifier_enabled = True


def _frame(rows: int = 120, close: float = 100.0, atr: float = 0.8, adx: float = 22.0, vol: float = 1000.0, flat: bool = False) -> pd.DataFrame:
    idx = pd.RangeIndex(rows)
    if flat:
        base = pd.Series([close for _ in range(rows)], index=idx)
    else:
        base = pd.Series([close + ((-1) ** i) * (i % 5) * 0.2 for i in range(rows)], index=idx)
    high_spread = 0.1 if flat else 0.3
    low_spread = 0.1 if flat else 0.3
    return pd.DataFrame(
        {
            "close": base,
            "open": base - 0.1,
            "high": base + high_spread,
            "low": base - low_spread,
            "volume": pd.Series([vol + (100 if i % 10 == 0 else 0) for i in range(rows)], index=idx),
            "atr": pd.Series([atr] * rows, index=idx),
            "adx": pd.Series([adx] * rows, index=idx),
            "ema20": base.rolling(20, min_periods=1).mean(),
        }
    )


def test_volatility_filter_blocks_low_volatility() -> None:
    df = _frame(atr=0.1, adx=10.0, vol=100, flat=True)
    result = VolatilityFilter().evaluate(df)
    assert result.state.value == "LOW_VOLATILITY"
    assert result.allow_trade is False


def test_liquidity_zone_detector_side_rules() -> None:
    df = _frame()
    detector = LiquidityZoneDetector(proximity_threshold=0.01)
    zones = detector.detect(df)
    assert zones.support_zone is not None
    assert zones.resistance_zone is not None
    price = float(df["close"].iloc[-1])
    assert detector.is_side_allowed("BUY", zones, price) or detector.is_side_allowed("SELL", zones, price)


def test_market_intelligence_returns_snapshot_fields() -> None:
    df = _frame()
    mi = MarketIntelligence(_Settings())
    result = mi.evaluate("BUY", {"frame5": df, "price": float(df["close"].iloc[-1])})
    assert "market_regime" in result
    assert "volatility_state" in result
    assert "distance_to_support" in result
    assert "distance_to_resistance" in result
    assert "filter_details" in result


def test_market_intelligence_applies_soft_multiplier_floor() -> None:
    df = _frame(atr=0.1, adx=10.0, vol=100, flat=True)
    mi = MarketIntelligence(_Settings())
    result = mi.evaluate("BUY", {"frame5": df, "price": float(df["close"].iloc[-1])})
    assert result["size_multiplier"] >= 0.35


def test_market_intelligence_uses_configurable_liquidity_threshold() -> None:
    class Tight:
        enable_market_intelligence = True
        volatility_filter_enabled = True
        liquidity_zone_filter_enabled = True
        market_regime_classifier_enabled = True
        market_range_filter_enabled = True
        liquidity_proximity_threshold = 0.0015

    class Loose(Tight):
        liquidity_proximity_threshold = 0.005

    df = _frame(close=100.0)
    price = float(df["close"].iloc[-1])

    tight = MarketIntelligence(Tight())
    loose = MarketIntelligence(Loose())

    tight_result = tight.evaluate("BUY", {"frame5": df, "price": price, "signal_confidence": 0.8})
    loose_result = loose.evaluate("BUY", {"frame5": df, "price": price, "signal_confidence": 0.8})

    assert loose_result["size_multiplier"] >= tight_result["size_multiplier"]
