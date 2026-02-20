from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class VolatilityDecision:
    allow_trading: bool
    exposure_multiplier: float
    reason: str


class ExtremeVolatilityFilter:
    def __init__(self, atr_explosion_threshold: float = 2.2, zscore_threshold: float = 2.5) -> None:
        self.atr_explosion_threshold = atr_explosion_threshold
        self.zscore_threshold = zscore_threshold

    def evaluate(self, frame: pd.DataFrame) -> VolatilityDecision:
        atr = frame["atr14"]
        if len(atr) < 30:
            return VolatilityDecision(True, 1.0, "insufficient_history")

        atr_ratio = float(atr.iloc[-1] / max(atr.rolling(20).mean().iloc[-1], 1e-9))
        vol = frame["volatility20"]
        vol_z = float((vol.iloc[-1] - vol.mean()) / (vol.std() + 1e-9))

        if atr_ratio >= self.atr_explosion_threshold and vol_z >= self.zscore_threshold:
            return VolatilityDecision(False, 0.0, "volatility_trading_halt")
        if atr_ratio >= self.atr_explosion_threshold or vol_z >= self.zscore_threshold:
            shrink = float(np.clip(1.0 / max(atr_ratio, 1.0), 0.2, 0.8))
            return VolatilityDecision(True, shrink, "volatility_size_reduction")
        return VolatilityDecision(True, 1.0, "volatility_normal")
