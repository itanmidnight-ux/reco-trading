from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd


class VolatilityState(str, Enum):
    LOW_VOLATILITY = "LOW_VOLATILITY"
    NORMAL_VOLATILITY = "NORMAL_VOLATILITY"
    EXTREME_VOLATILITY = "EXTREME_VOLATILITY"


class MarketRegime(str, Enum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_ACTIVITY = "LOW_ACTIVITY"


@dataclass(slots=True)
class VolatilityAssessment:
    state: VolatilityState
    allow_trade: bool
    risk_multiplier: float


@dataclass(slots=True)
class LiquidityZones:
    support_zone: float | None
    resistance_zone: float | None
    distance_to_support: float | None
    distance_to_resistance: float | None


@dataclass(slots=True)
class MarketRegimeAssessment:
    regime: MarketRegime
    allow_trade: bool
    risk_multiplier: float


class VolatilityFilter:
    """Classifies volatility and proposes risk adjustment."""

    def evaluate(self, df: pd.DataFrame) -> VolatilityAssessment:
        if len(df) < 30:
            return VolatilityAssessment(VolatilityState.NORMAL_VOLATILITY, allow_trade=True, risk_multiplier=1.0)

        recent = df.tail(30)
        close = float(recent["close"].iloc[-1])
        atr = float(recent["atr"].iloc[-1])
        adx = float(recent.get("adx", pd.Series([20.0])).iloc[-1])

        candle_range_pct = ((recent["high"] - recent["low"]) / recent["close"].clip(lower=1e-9)).mean()
        atr_ratio = atr / max(close, 1e-9)
        recent_vol = recent["close"].pct_change().dropna().std()

        if atr_ratio < 0.003 and candle_range_pct < 0.004 and recent_vol < 0.004 and adx < 16:
            return VolatilityAssessment(VolatilityState.LOW_VOLATILITY, allow_trade=False, risk_multiplier=0.0)
        if atr_ratio > 0.02 or candle_range_pct > 0.025 or recent_vol > 0.02:
            return VolatilityAssessment(VolatilityState.EXTREME_VOLATILITY, allow_trade=True, risk_multiplier=0.6)
        return VolatilityAssessment(VolatilityState.NORMAL_VOLATILITY, allow_trade=True, risk_multiplier=1.0)


class LiquidityZoneDetector:
    """Approximates support/resistance zones from swings, clustering and volume."""

    def __init__(self, proximity_threshold: float = 0.0025) -> None:
        self.proximity_threshold = proximity_threshold

    def detect(self, df: pd.DataFrame) -> LiquidityZones:
        if len(df) < 20:
            return LiquidityZones(None, None, None, None)

        recent = df.tail(80).copy()
        vol_mean = recent["volume"].rolling(20).mean()
        volume_spikes = recent["volume"] > (vol_mean * 1.4)

        swing_highs = recent[(recent["high"] == recent["high"].rolling(5, center=True).max()) | volume_spikes]
        swing_lows = recent[(recent["low"] == recent["low"].rolling(5, center=True).min()) | volume_spikes]

        support_zone = self._cluster_price(swing_lows.get("low"), float(recent["close"].iloc[-1]), prefer="lower")
        resistance_zone = self._cluster_price(swing_highs.get("high"), float(recent["close"].iloc[-1]), prefer="upper")

        price = float(recent["close"].iloc[-1])
        distance_to_support = self._distance_ratio(price, support_zone)
        distance_to_resistance = self._distance_ratio(price, resistance_zone)

        return LiquidityZones(
            support_zone=support_zone,
            resistance_zone=resistance_zone,
            distance_to_support=distance_to_support,
            distance_to_resistance=distance_to_resistance,
        )

    def is_side_allowed(self, side: str, zones: LiquidityZones, price: float) -> bool:
        if side == "BUY":
            return self._distance_ratio(price, zones.support_zone) is not None and self._distance_ratio(
                price, zones.support_zone
            ) <= self.proximity_threshold
        if side == "SELL":
            return self._distance_ratio(price, zones.resistance_zone) is not None and self._distance_ratio(
                price, zones.resistance_zone
            ) <= self.proximity_threshold
        return False

    def _cluster_price(self, series: pd.Series | None, price: float, prefer: str) -> float | None:
        if series is None:
            return None
        cleaned = pd.to_numeric(series, errors="coerce").dropna()
        if cleaned.empty:
            return None

        rounded = (cleaned / max(price, 1e-9) * 1000).round() / 1000
        bins = rounded.value_counts()
        if bins.empty:
            return None

        top_levels = bins.head(3).index.tolist()
        levels = [float(level * price) for level in top_levels]
        if prefer == "lower":
            lower = [lvl for lvl in levels if lvl <= price]
            return max(lower) if lower else min(levels)
        upper = [lvl for lvl in levels if lvl >= price]
        return min(upper) if upper else max(levels)

    def _distance_ratio(self, price: float, zone: float | None) -> float | None:
        if zone is None:
            return None
        return abs(price - zone) / max(price, 1e-9)


class MarketRegimeClassifier:
    """Classifies market regime from trend/volatility descriptors."""

    def classify(self, df: pd.DataFrame) -> MarketRegimeAssessment:
        if len(df) < 50:
            return MarketRegimeAssessment(MarketRegime.RANGING, allow_trade=True, risk_multiplier=1.0)

        recent = df.tail(100)
        price = float(recent["close"].iloc[-1])
        atr_ratio = float(recent["atr"].iloc[-1]) / max(price, 1e-9)
        adx = float(recent.get("adx", pd.Series([20.0])).iloc[-1])

        ema20 = recent["ema20"]
        slope = (float(ema20.iloc[-1]) - float(ema20.iloc[-10])) / max(price, 1e-9)
        returns = recent["close"].pct_change().dropna().abs()
        vol_now = float(returns.iloc[-1]) if not returns.empty else 0.0
        vol_pct = float((returns <= vol_now).mean()) if not returns.empty else 0.5

        if atr_ratio < 0.002 and adx < 14 and vol_pct < 0.35:
            return MarketRegimeAssessment(MarketRegime.LOW_ACTIVITY, allow_trade=False, risk_multiplier=0.0)
        if atr_ratio > 0.02 or vol_pct > 0.92:
            return MarketRegimeAssessment(MarketRegime.HIGH_VOLATILITY, allow_trade=True, risk_multiplier=0.5)
        if adx >= 25 and abs(slope) > 0.0015:
            return MarketRegimeAssessment(MarketRegime.TRENDING, allow_trade=True, risk_multiplier=1.0)
        return MarketRegimeAssessment(MarketRegime.RANGING, allow_trade=True, risk_multiplier=0.85)


class MarketIntelligence:
    """Coordinator that keeps advanced filters additive and optional."""

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.volatility_filter = VolatilityFilter()
        self.liquidity_detector = LiquidityZoneDetector(proximity_threshold=0.0025)
        self.regime_classifier = MarketRegimeClassifier()

    def evaluate(self, side: str, market_data: dict[str, Any]) -> dict[str, Any]:
        if not getattr(self.settings, "enable_market_intelligence", True):
            return {"approved": True, "size_multiplier": 1.0, "reason": "DISABLED"}

        df = market_data.get("frame5")
        price = float(market_data.get("price", 0.0))
        result = {
            "approved": True,
            "size_multiplier": 1.0,
            "reason": "APPROVED",
            "market_regime": None,
            "volatility_state": None,
            "distance_to_support": None,
            "distance_to_resistance": None,
            "support_zone": None,
            "resistance_zone": None,
        }

        if df is None or len(df) == 0:
            return result

        if getattr(self.settings, "volatility_filter_enabled", True):
            vol = self.volatility_filter.evaluate(df)
            result["volatility_state"] = vol.state.value
            result["size_multiplier"] *= vol.risk_multiplier
            if not vol.allow_trade:
                result["approved"] = False
                result["reason"] = vol.state.value

        if getattr(self.settings, "liquidity_zone_filter_enabled", True):
            zones = self.liquidity_detector.detect(df)
            result["support_zone"] = zones.support_zone
            result["resistance_zone"] = zones.resistance_zone
            result["distance_to_support"] = zones.distance_to_support
            result["distance_to_resistance"] = zones.distance_to_resistance
            if result["approved"] and not self.liquidity_detector.is_side_allowed(side, zones, price):
                result["approved"] = False
                result["reason"] = "LIQUIDITY_ZONE_FILTER"

        if getattr(self.settings, "market_regime_classifier_enabled", True):
            regime = self.regime_classifier.classify(df)
            result["market_regime"] = regime.regime.value
            result["size_multiplier"] *= regime.risk_multiplier
            if not regime.allow_trade:
                result["approved"] = False
                result["reason"] = regime.regime.value

        result["size_multiplier"] = max(min(float(result["size_multiplier"]), 1.0), 0.1)
        return result
