from __future__ import annotations

import numpy as np
import pandas as pd


class FactorEvaluation:
    @staticmethod
    def ic(factor_values: pd.Series, future_returns: pd.Series) -> float:
        aligned = pd.concat([factor_values, future_returns], axis=1).dropna()
        if aligned.empty:
            return 0.0
        return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method='spearman'))

    @staticmethod
    def ic_decay(factor_values: pd.Series, future_returns: pd.Series, horizons: list[int]) -> dict[int, float]:
        decay: dict[int, float] = {}
        for horizon in horizons:
            shifted = future_returns.shift(-horizon)
            decay[horizon] = FactorEvaluation.ic(factor_values, shifted)
        return decay

    @staticmethod
    def turnover(factor_values: pd.Series) -> float:
        signal = np.sign(factor_values.fillna(0.0))
        return float(signal.diff().abs().mean())

    @staticmethod
    def factor_correlation(factors: pd.DataFrame) -> pd.DataFrame:
        return factors.corr()

    @staticmethod
    def orthogonalize(target_factor: pd.Series, against_factors: pd.DataFrame) -> pd.Series:
        design = against_factors.copy().fillna(0.0)
        design['bias'] = 1.0
        valid = pd.concat([target_factor, design], axis=1).dropna()
        if valid.empty:
            return target_factor * 0
        y = valid.iloc[:, 0].to_numpy(dtype=float)
        x = valid.iloc[:, 1:].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        residual = y - x @ beta
        result = pd.Series(index=target_factor.index, dtype=float)
        result.loc[valid.index] = residual
        return result

    @staticmethod
    def neutralize(factor_values: pd.Series, exposures: pd.DataFrame) -> pd.Series:
        return FactorEvaluation.orthogonalize(factor_values, exposures)
