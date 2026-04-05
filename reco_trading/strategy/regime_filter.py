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
        try:
            atr_val = float(row["atr"])
            close_val = float(row["close"])
            if close_val <= 0:
                return RegimeDecision(VolatilityRegime.NORMAL_VOLATILITY, 0.0, True, 1.0)
            atr_ratio = atr_val / close_val
        except (KeyError, TypeError, ZeroDivisionError):
            return RegimeDecision(VolatilityRegime.NORMAL_VOLATILITY, 0.0, True, 1.0)

        if atr_ratio < self.low_threshold:
            # Mercado dormido: bajo ATR = sin momentum = no operar
            return RegimeDecision(VolatilityRegime.LOW_VOLATILITY, atr_ratio, allow_trade=False, size_multiplier=0.0)
        if atr_ratio > self.high_threshold:
            # Alta volatilidad: operar con tamaño reducido como protección
            return RegimeDecision(VolatilityRegime.HIGH_VOLATILITY, atr_ratio, allow_trade=True, size_multiplier=0.60)
        # Régimen normal: operación completa
        return RegimeDecision(VolatilityRegime.NORMAL_VOLATILITY, atr_ratio, allow_trade=True, size_multiplier=1.0)
