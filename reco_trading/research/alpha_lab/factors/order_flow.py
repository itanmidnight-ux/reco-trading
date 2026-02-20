from __future__ import annotations

import pandas as pd

from reco_trading.research.alpha_lab.base import AlphaFactor


class OrderFlowFactor(AlphaFactor):
    def compute(self, frame: pd.DataFrame) -> pd.Series:
        self.validate(frame)
        if 'buy_volume' not in frame.columns or 'sell_volume' not in frame.columns:
            raise ValueError(f'{self.name}: required columns "buy_volume" and "sell_volume" are missing')
        imbalance = (frame['buy_volume'] - frame['sell_volume']) / (frame['buy_volume'] + frame['sell_volume']).replace(0, pd.NA)
        return imbalance.rolling(self.lookback).mean()
