from __future__ import annotations

import numpy as np
import pandas as pd


class PortfolioBuilder:
    @staticmethod
    def rank_weighting(factor_values: pd.Series) -> pd.Series:
        ranked = factor_values.rank(pct=True)
        centered = ranked - ranked.mean()
        denom = centered.abs().sum()
        return centered / denom if denom > 0 else centered

    @staticmethod
    def zscore_normalization(factor_values: pd.Series) -> pd.Series:
        std = factor_values.std(ddof=0)
        if std == 0 or pd.isna(std):
            return factor_values * 0
        return (factor_values - factor_values.mean()) / std

    @staticmethod
    def risk_parity_by_factors(factor_returns: pd.DataFrame) -> pd.Series:
        vol = factor_returns.std(ddof=0).replace(0, np.nan)
        inv_vol = 1.0 / vol
        weights = inv_vol / inv_vol.sum()
        return weights.fillna(0.0)

    @staticmethod
    def ic_weighted_allocation(factor_ics: dict[str, float]) -> dict[str, float]:
        positive = {name: max(ic, 0.0) for name, ic in factor_ics.items()}
        total = sum(positive.values())
        if total <= 0:
            n = len(factor_ics)
            return {name: 1 / n for name in factor_ics} if n > 0 else {}
        return {name: value / total for name, value in positive.items()}
