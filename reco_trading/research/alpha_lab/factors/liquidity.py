from __future__ import annotations

import pandas as pd

from reco_trading.research.alpha_lab.base import AlphaFactor


class LiquidityFactor(AlphaFactor):
    def compute(self, frame: pd.DataFrame) -> pd.Series:
        self.validate(frame)
        if 'volume' not in frame.columns:
            raise ValueError(f'{self.name}: required column "volume" is missing')
        dollar_volume = frame['close'] * frame['volume']
        return dollar_volume.rolling(self.lookback).mean().pct_change()
