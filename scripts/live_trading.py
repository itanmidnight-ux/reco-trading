from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pandas as pd

from reco_trading.core.feature_engine import FeatureEngine
from reco_trading.core.mean_reversion_model import MeanReversionModel
from reco_trading.core.momentum_model import MomentumModel
from reco_trading.kernel.quant_kernel import QuantKernel


@dataclass(slots=True)
class RuntimeState:
    equity: float
    daily_pnl: float
    consecutive_losses: int


class FeatureEngineAdapter:
    def __init__(
        self,
        feature_engine: FeatureEngine,
        momentum_model: MomentumModel,
        mean_reversion_model: MeanReversionModel,
        runtime_state: RuntimeState,
    ) -> None:
        self._feature_engine = feature_engine
        self._momentum_model = momentum_model
        self._mean_reversion_model = mean_reversion_model
        self._runtime_state = runtime_state

    def compute(self, ohlcv: pd.DataFrame) -> dict[str, Any]:
        feats = self._feature_engine.build(ohlcv)
        momentum = float(self._momentum_model.predict_proba_up(feats))
        mean_reversion = float(self._mean_reversion_model.predict_reversion(feats))
        returns = feats['return'].tail(300).to_numpy(dtype=float)
        atr = float(feats.iloc[-1]['atr14'])
        volatility = float(feats['return'].tail(50).std() or 0.0)
        win_rate = 0.55 if self._runtime_state.consecutive_losses == 0 else 0.5
        return {
            'returns': returns,
            'returns_df': pd.DataFrame({'BTCUSDT': returns}),
            'prices': feats['close'].tail(300),
            'signals': {'momentum': momentum, 'mean_reversion': mean_reversion},
            'volatility': volatility,
            'equity': float(self._runtime_state.equity),
            'atr': atr,
            'win_rate': win_rate,
            'reward_risk': 1.8,
        }


class FusionEngineAdapter:
    def fuse(self, signals: dict[str, float], regime: str, volatility: float) -> float:
        momentum = float(signals.get('momentum', 0.5))
        mean_reversion = float(signals.get('mean_reversion', 0.5))
        base = 0.65 * momentum + 0.35 * (1.0 - mean_reversion)
        if regime == 'volatile':
            base *= max(0.4, 1.0 - min(max(volatility, 0.0), 1.0))
        return max(0.0, min(1.0, base))


if __name__ == '__main__':
    asyncio.run(QuantKernel().run())
