from __future__ import annotations

import pandas as pd

from reco_trading.strategy.market_intelligence import (
    LiquidityZoneDetector,
    MarketIntelligence,
    MarketRegime,
    MarketRegimeClassifier,
    VolatilityFilter,
)


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



def _frame_with_adx(adx_values: list[float], slope: float = 0.002) -> pd.DataFrame:
    rows = len(adx_values)
    idx = pd.RangeIndex(rows)
    closes = pd.Series([100.0 + i * 0.05 for i in range(rows)], index=idx)
    ema20 = pd.Series([100.0 + i * slope for i in range(rows)], index=idx)
    return pd.DataFrame(
        {
            "close": closes,
            "open": closes - 0.1,
            "high": closes + 0.3,
            "low": closes - 0.3,
            "volume": pd.Series([1000.0] * rows, index=idx),
            "atr": pd.Series([0.8] * rows, index=idx),
            "adx": pd.Series(adx_values, index=idx),
            "ema20": ema20,
        }
    )


def test_liquidity_dbscan_dense_cluster() -> None:
    """Verifica que DBSCAN agrupa swing highs correctamente."""
    df = pd.DataFrame(
        {
            "high": [49900, 49950, 50000, 50050, 50100] + [50000] * 10 + [49500] * 10,
            "low": [48000] * 25,
            "volume": [1000] * 25,
            "atr": [100] * 25,
            "close": [50000] * 25,
        }
    )

    detector = LiquidityZoneDetector()
    zones = detector.detect(df)

    assert zones.resistance_zone is not None
    assert 49750 < zones.resistance_zone < 50250


def test_market_regime_tracks_transitions() -> None:
    classifier = MarketRegimeClassifier()

    df1 = _frame_with_adx([15.0] * 95 + [30.0] * 5, slope=0.25)
    result1 = classifier.classify(df1)

    assert hasattr(classifier, "_last_regime")
    assert hasattr(classifier, "_regime_change_candles")
    assert result1.regime == MarketRegime.TRENDING
    assert 0.35 <= result1.risk_multiplier <= 1.0

    initial_candles = classifier._regime_change_candles
    df2 = _frame_with_adx([25.0] * 100, slope=0.2)
    classifier.classify(df2)
    assert classifier._regime_change_candles == max(initial_candles - 1, 0)


def test_market_regime_mid_trend_uses_expected_multiplier() -> None:
    classifier = MarketRegimeClassifier()
    df = _frame_with_adx([22.0] * 100, slope=0.15)

    result = classifier.classify(df)

    assert result.regime == MarketRegime.TRENDING
    assert result.risk_multiplier == 0.8
