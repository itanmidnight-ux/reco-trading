from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class VolatilityRegime(str, Enum):
    LOW_VOLATILITY = "LOW_VOLATILITY"
    NORMAL_VOLATILITY = "NORMAL_VOLATILITY"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"


@dataclass(slots=True)
class RegimeDecision:
    regime: VolatilityRegime
    atr_ratio: float
    allow_trade: bool
    size_multiplier: float


class RegimeFilter:
    """Classifies volatility regime using ATR ratio, ADX and volatility percentile."""

    def __init__(self, low_threshold: float = 0.0025, high_threshold: float = 0.015) -> None:
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def evaluate(self, frame: pd.DataFrame) -> RegimeDecision:
        recent = frame.tail(120)
        row = recent.iloc[-1]
        close = max(float(row["close"]), 1e-9)
        atr_ratio = float(row["atr"] / close)
        adx = float(row.get("adx", 0.0))

        atr_series = (recent["atr"] / recent["close"].clip(lower=1e-9)).dropna()
        vol_percentile = float((atr_series <= atr_ratio).mean()) if not atr_series.empty else 0.5

        is_low = atr_ratio < self.low_threshold or adx < 15 or vol_percentile < 0.25
        is_high = atr_ratio > self.high_threshold or adx > 35 or vol_percentile > 0.85

        if is_low:
            return RegimeDecision(VolatilityRegime.LOW_VOLATILITY, atr_ratio, False, 0.0)
        if is_high:
            return RegimeDecision(VolatilityRegime.HIGH_VOLATILITY, atr_ratio, True, 0.6)
        return RegimeDecision(VolatilityRegime.NORMAL_VOLATILITY, atr_ratio, True, 1.0)
