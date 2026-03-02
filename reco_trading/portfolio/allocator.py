from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


class PortfolioAllocator:
    def __init__(
        self,
        max_weight_per_asset: float = 0.5,
        correlation_threshold: float = 0.8,
        correlation_penalty_strength: float = 0.5,
    ) -> None:
        self.max_weight_per_asset = float(np.clip(max_weight_per_asset, 0.01, 1.0))
        self.correlation_threshold = float(np.clip(correlation_threshold, 0.0, 1.0))
        self.correlation_penalty_strength = float(np.clip(correlation_penalty_strength, 0.0, 1.0))

    def _safe_fallback(self, assets: list[str]) -> dict[str, float]:
        if not assets:
            return {}
        base = 1.0 / float(len(assets))
        return {asset: base for asset in assets}

    def allocate(
        self,
        asset_edges: Mapping[str, float],
        asset_volatility: Mapping[str, float],
        correlation_matrix: pd.DataFrame | None,
    ) -> dict[str, float]:
        assets = [str(a) for a in asset_edges.keys()]
        if not assets:
            return {}
        if len(assets) == 1:
            return {assets[0]: 1.0}

        scores = {asset: float(max(asset_edges.get(asset, 0.0), 0.0)) for asset in assets}
        score_sum = float(sum(scores.values()))
        if score_sum <= 0.0:
            return self._safe_fallback(assets)

        weights = {asset: float(scores[asset] / score_sum) for asset in assets}

        if correlation_matrix is not None and not correlation_matrix.empty:
            corr = correlation_matrix.copy()
            for asset in assets:
                if asset not in corr.index or asset not in corr.columns:
                    continue
                penalty_multiplier = 1.0
                for other in assets:
                    if other == asset or other not in corr.index or other not in corr.columns:
                        continue
                    corr_ij = float(corr.loc[asset, other])
                    if np.isfinite(corr_ij) and corr_ij > self.correlation_threshold:
                        penalty_multiplier *= (1.0 - (self.correlation_penalty_strength * corr_ij))
                weights[asset] *= float(np.clip(penalty_multiplier, 0.05, 1.0))

        for asset in assets:
            _ = asset_volatility.get(asset, 0.0)
            weights[asset] = float(min(max(weights[asset], 0.0), self.max_weight_per_asset))

        total = float(sum(weights.values()))
        if total <= 0.0:
            return self._safe_fallback(assets)
        return {asset: float(weights[asset] / total) for asset in assets}
