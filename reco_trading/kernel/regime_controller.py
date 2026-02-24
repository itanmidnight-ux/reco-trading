from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class RegimeSnapshot:
    current_regime: str
    regime_stability_score: float


class RegimeController:
    def __init__(self, window: int = 120, stability_window: int = 40) -> None:
        self.vol_history: deque[float] = deque(maxlen=max(window, 30))
        self.spread_history: deque[float] = deque(maxlen=max(window, 30))
        self.liquidity_history: deque[float] = deque(maxlen=max(window, 30))
        self.regime_history: deque[str] = deque(maxlen=max(stability_window, 20))

    @staticmethod
    def _sanitize(value: float, default: float = 0.0) -> float:
        cast = float(value)
        return cast if np.isfinite(cast) else default

    def update(
        self,
        *,
        volatility: float,
        autocorr: float,
        avg_spread_bps: float,
        relative_liquidity: float,
    ) -> RegimeSnapshot:
        vol = self._sanitize(volatility)
        ac = float(np.clip(self._sanitize(autocorr), -1.0, 1.0))
        spread = max(self._sanitize(avg_spread_bps), 0.0)
        liquidity = max(self._sanitize(relative_liquidity, 1.0), 1e-6)

        self.vol_history.append(vol)
        self.spread_history.append(spread)
        self.liquidity_history.append(liquidity)

        vol_pct = float(np.mean(np.asarray(self.vol_history, dtype=float) <= vol))
        spread_pct = float(np.mean(np.asarray(self.spread_history, dtype=float) <= spread))

        if liquidity < 0.55 or spread_pct > 0.90:
            regime = 'LIQUIDITY_STRESS'
        elif vol_pct >= 0.80:
            regime = 'HIGH_VOL_REGIME'
        elif vol_pct <= 0.20:
            regime = 'LOW_VOL_REGIME'
        elif ac >= 0.15:
            regime = 'TREND_REGIME'
        else:
            regime = 'MEAN_REVERT_REGIME'

        self.regime_history.append(regime)
        stable = sum(1 for r in self.regime_history if r == regime)
        stability = float(np.clip(stable / max(len(self.regime_history), 1), 0.0, 1.0))
        return RegimeSnapshot(current_regime=regime, regime_stability_score=stability)
