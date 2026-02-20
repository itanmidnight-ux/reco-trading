from __future__ import annotations

import pandas as pd

from reco_trading.research.alpha_lab.base import AlphaFactor


class MomentumFactor(AlphaFactor):
    def compute(self, frame: pd.DataFrame) -> pd.Series:
        self.validate(frame)
        return frame['close'].pct_change(self.lookback)
