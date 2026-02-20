from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

try:
    from hmmlearn.hmm import GaussianHMM
except Exception:  # pragma: no cover
    GaussianHMM = None


class MarketRegimeDetector:
    def __init__(self, n_states: int = 3):
        self.hmm = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=200) if GaussianHMM else None
        self.gmm = GaussianMixture(n_components=n_states)
        self.fitted = False

    @staticmethod
    def _as_2d(returns: np.ndarray) -> np.ndarray:
        arr = np.asarray(returns, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def fit(self, returns: np.ndarray) -> None:
        x = self._as_2d(returns)
        if self.hmm is not None:
            self.hmm.fit(x)
        self.gmm.fit(x)
        self.fitted = True

    def detect_volatility_state(self, returns: pd.Series) -> str:
        vol = returns.rolling(20).std().iloc[-1]
        z_score = (vol - returns.std()) / (returns.std() + 1e-8)

        if z_score > 2:
            return 'high_volatility'
        if z_score < -1:
            return 'low_volatility'
        return 'normal'

    def detect_trend(self, prices: pd.Series) -> str:
        ma_fast = prices.rolling(20).mean().iloc[-1]
        ma_slow = prices.rolling(50).mean().iloc[-1]

        if ma_fast > ma_slow:
            return 'trend'
        return 'range'

    def predict(self, returns: np.ndarray, prices: pd.Series):
        x = self._as_2d(returns)

        if not self.fitted:
            self.fit(x)

        if self.hmm is not None:
            hidden_states = self.hmm.predict(x)
            current_state = int(hidden_states[-1])
            confidence = float(np.max(self.hmm.predict_proba(x)[-1]))
        else:
            probs = self.gmm.predict_proba(x)
            current_state = int(np.argmax(probs[-1]))
            confidence = float(np.max(probs[-1]))

        vol_state = self.detect_volatility_state(pd.Series(x.flatten()))
        trend_state = self.detect_trend(prices)

        regime_map = {0: 'range', 1: 'trend', 2: 'high_volatility'}

        return {
            'regime': regime_map.get(current_state, 'range'),
            'volatility_state': vol_state,
            'trend_state': trend_state,
            'confidence': confidence,
        }
