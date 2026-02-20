from __future__ import annotations

import pandas as pd

from reco_trading.research.alpha_lab.base import AlphaFactor


class VolatilityFactor(AlphaFactor):
    def compute(self, frame: pd.DataFrame) -> pd.Series:
        self.validate(frame)
        returns = frame['close'].pct_change()
        return -returns.rolling(self.lookback).std(ddof=0)
