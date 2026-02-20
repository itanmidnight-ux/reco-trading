from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

try:
    from hmmlearn.hmm import GaussianHMM
except Exception:  # pragma: no cover - fallback si dependencia no está instalada
    GaussianHMM = None


@dataclass(slots=True)
class RegimeState:
    regime: str
    confidence: float
    volatility_state: str


class MarketRegimeDetector:
    def __init__(self, n_states: int = 3, persistence: float = 0.7) -> None:
        self.n_states = n_states
        self.persistence = float(np.clip(persistence, 0.0, 0.99))
        self._prev_regime: str = "range"

    def _build_features(self, frame: pd.DataFrame) -> np.ndarray:
        rets = frame["return"].to_numpy(dtype=float)
        vol = frame["volatility20"].to_numpy(dtype=float)
        trend = (frame["ema12"] - frame["ema26"]).to_numpy(dtype=float)
        return np.column_stack([rets, vol, trend])

    def _infer_state(self, x: np.ndarray) -> tuple[int, float]:
        if x.shape[0] < 30:
            return 0, 0.4

        if GaussianHMM is not None:
            hmm = GaussianHMM(n_components=self.n_states, covariance_type="diag", n_iter=150, random_state=7)
            hmm.fit(x)
            state_seq = hmm.predict(x)
            state = int(state_seq[-1])
            post = hmm.predict_proba(x)[-1]
            conf = float(np.max(post))
            return state, conf

        gmm = GaussianMixture(n_components=self.n_states, covariance_type="full", random_state=7)
        gmm.fit(x)
        probs = gmm.predict_proba(x)[-1]
        return int(np.argmax(probs)), float(np.max(probs))

    def detect(self, frame: pd.DataFrame) -> RegimeState:
        x = self._build_features(frame)
        state, conf = self._infer_state(x)

        last = frame.iloc[-1]
        trend_strength = abs(float(last["ema12"] - last["ema26"])) / max(float(last["close"]), 1e-9)
        vol = float(last["volatility20"])
        vol_z = (frame["volatility20"].iloc[-1] - frame["volatility20"].mean()) / (frame["volatility20"].std() + 1e-9)

        if vol_z > 2.0 or vol > frame["volatility20"].quantile(0.90):
            regime = "high_volatility"
            vol_state = "explosive"
        elif trend_strength > 0.0025 and state != 1:
            regime = "trend"
            vol_state = "normal"
        else:
            regime = "range"
            vol_state = "compressed" if vol_z < -1.0 else "normal"

        if regime != self._prev_regime and conf < self.persistence:
            regime = self._prev_regime
            conf *= 0.8
        else:
            self._prev_regime = regime

        return RegimeState(regime=regime, confidence=float(np.clip(conf, 0.0, 1.0)), volatility_state=vol_state)
