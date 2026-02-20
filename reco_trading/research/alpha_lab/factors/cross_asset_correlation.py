from __future__ import annotations

import pandas as pd

from reco_trading.research.alpha_lab.base import AlphaFactor


class CrossAssetCorrelationFactor(AlphaFactor):
    def compute(self, frame: pd.DataFrame) -> pd.Series:
        self.validate(frame)
        if 'benchmark_close' not in frame.columns:
            raise ValueError(f'{self.name}: required column "benchmark_close" is missing')
        asset_ret = frame['close'].pct_change()
        benchmark_ret = frame['benchmark_close'].pct_change()
        rolling_corr = asset_ret.rolling(self.lookback).corr(benchmark_ret)
        return -rolling_corr
