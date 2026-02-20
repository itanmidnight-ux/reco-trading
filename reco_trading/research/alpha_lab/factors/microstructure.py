from __future__ import annotations

import pandas as pd

from reco_trading.research.alpha_lab.base import AlphaFactor


class MicrostructureFactor(AlphaFactor):
    def compute(self, frame: pd.DataFrame) -> pd.Series:
        self.validate(frame)
        if 'high' not in frame.columns or 'low' not in frame.columns:
            raise ValueError(f'{self.name}: required columns "high" and "low" are missing')
        spread_proxy = (frame['high'] - frame['low']) / frame['close'].replace(0, pd.NA)
        return -spread_proxy.rolling(self.lookback).mean()
