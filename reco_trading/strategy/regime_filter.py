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
    """Classifies volatility regime using ATR/price ratio. Optimized for crypto markets."""

    def __init__(self, low_threshold: float = 0.003, high_threshold: float = 0.018) -> None:
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def evaluate(self, frame: pd.DataFrame) -> RegimeDecision:
        row = frame.iloc[-1]
        atr_ratio = float(row["atr"] / row["close"])

        if atr_ratio < self.low_threshold:
            return RegimeDecision(VolatilityRegime.LOW_VOLATILITY, atr_ratio, True, 0.50)
        if atr_ratio > self.high_threshold:
            return RegimeDecision(VolatilityRegime.HIGH_VOLATILITY, atr_ratio, True, 0.65)
        return RegimeDecision(VolatilityRegime.NORMAL_VOLATILITY, atr_ratio, True, 1.0)
