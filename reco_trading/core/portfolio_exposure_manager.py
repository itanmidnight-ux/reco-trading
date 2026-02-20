from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ExposureSnapshot:
    correlation_matrix: pd.DataFrame
    concentration_risk: float
    heatmap: dict[str, float]
    exchange_notionals: dict[str, float]
    cross_exchange_notional: float


class PortfolioExposureManager:
    def __init__(self, lookback: int = 200) -> None:
        self.lookback = lookback

    @staticmethod
    def _exchange_from_key(asset_key: str) -> str:
        if ':' not in asset_key:
            return 'global'
        return asset_key.split(':', 1)[0]

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

        exchange_notionals: dict[str, float] = {}
        for asset_key, notional in notionals.items():
            exchange = self._exchange_from_key(asset_key)
            exchange_notionals[exchange] = exchange_notionals.get(exchange, 0.0) + float(notional)

        cross_exchange_notional = float(sum(abs(v) for v in exchange_notionals.values()))

        return ExposureSnapshot(
            correlation_matrix=corr,
            concentration_risk=concentration,
            heatmap=heatmap,
            exchange_notionals=exchange_notionals,
            cross_exchange_notional=cross_exchange_notional,
        )

    def risk_parity_weights(self, returns: pd.DataFrame) -> dict[str, float]:
        vol = returns.std().replace(0.0, np.nan)
        inv_vol = 1.0 / vol
        weights = (inv_vol / inv_vol.sum()).fillna(0.0)
        return {k: float(v) for k, v in weights.items()}
