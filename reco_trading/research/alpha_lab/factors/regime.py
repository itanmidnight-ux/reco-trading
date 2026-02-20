from __future__ import annotations

import pandas as pd

from reco_trading.research.alpha_lab.base import AlphaFactor


class RegimeFactor(AlphaFactor):
    def compute(self, frame: pd.DataFrame) -> pd.Series:
        self.validate(frame)
        trend = frame['close'].rolling(self.lookback).mean()
        slow = frame['close'].rolling(self.lookback * 2).mean()
        return (trend - slow) / slow.replace(0, pd.NA)
