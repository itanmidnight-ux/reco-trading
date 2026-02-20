from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ExposureSnapshot:
    correlation_matrix: pd.DataFrame
    concentration_risk: float
    heatmap: dict[str, float]


class PortfolioExposureManager:
    def __init__(self, lookback: int = 200) -> None:
        self.lookback = lookback

    def evaluate(self, returns: pd.DataFrame, notionals: dict[str, float]) -> ExposureSnapshot:
        ret = returns.tail(self.lookback).copy()
        corr = ret.corr().fillna(0.0)

        if corr.shape[0] > 1:
            eigvals = np.linalg.eigvalsh(corr.to_numpy())
            concentration = float(np.max(eigvals) / max(np.sum(eigvals), 1e-9))
        else:
            concentration = 0.0

        total = max(sum(abs(v) for v in notionals.values()), 1e-9)
        heatmap = {asset: float(abs(notional) / total) for asset, notional in notionals.items()}

        return ExposureSnapshot(correlation_matrix=corr, concentration_risk=concentration, heatmap=heatmap)

    def risk_parity_weights(self, returns: pd.DataFrame) -> dict[str, float]:
        vol = returns.std().replace(0.0, np.nan)
        inv_vol = 1.0 / vol
        weights = (inv_vol / inv_vol.sum()).fillna(0.0)
        return {k: float(v) for k, v in weights.items()}
