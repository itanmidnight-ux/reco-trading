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
    """Classifies volatility regime using ATR/price ratio."""

    def __init__(self, low_threshold: float = 0.0025, high_threshold: float = 0.015) -> None:
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def evaluate(self, frame: pd.DataFrame) -> RegimeDecision:
        if frame.empty:
            return RegimeDecision(VolatilityRegime.LOW_VOLATILITY, 0.0, False, 0.0)

        row = frame.iloc[-1]
        atr_ratio = float(row["atr"] / row["close"])

        if atr_ratio < self.low_threshold:
            return RegimeDecision(VolatilityRegime.LOW_VOLATILITY, atr_ratio, False, 0.0)
        if atr_ratio > self.high_threshold:
            return RegimeDecision(VolatilityRegime.HIGH_VOLATILITY, atr_ratio, True, 0.7)
        return RegimeDecision(VolatilityRegime.NORMAL_VOLATILITY, atr_ratio, True, 1.0)
