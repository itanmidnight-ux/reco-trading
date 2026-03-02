from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


class ExposureModel:
    def __init__(self, lookback_bars: int = 100) -> None:
        self.lookback_bars = int(max(lookback_bars, 10))

    def compute_correlation_matrix(self, price_history_dict: Mapping[str, pd.Series]) -> pd.DataFrame:
        if not price_history_dict:
            return pd.DataFrame()

        returns_map: dict[str, pd.Series] = {}
        for asset, prices in price_history_dict.items():
            series = pd.Series(prices).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
            if series.size < 3:
                continue
            trimmed = series.tail(self.lookback_bars)
            log_returns = np.log(trimmed / trimmed.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
            if not log_returns.empty:
                returns_map[str(asset)] = log_returns

        if not returns_map:
            return pd.DataFrame()

        returns_df = pd.DataFrame(returns_map).dropna(how='all')
        if returns_df.empty:
            return pd.DataFrame()
        return returns_df.corr().fillna(0.0)
