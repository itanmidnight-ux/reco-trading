from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class MomentumModel:
    FEATURES = ['return', 'ema12', 'ema26', 'macd', 'breakout20', 'volatility20', 'volume_norm']

    def __init__(self) -> None:
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=0.3, max_iter=2000, class_weight='balanced')),
        ])
        self._fitted = False

    def fit(self, frame: pd.DataFrame) -> None:
        target = frame['target_up']
        if target.nunique(dropna=True) < 2:
            # Fallback defensivo para ventanas con una única clase.
            # Evita que el kernel entre en estado ERROR por ValueError de sklearn.
            self._fitted = False
            return
        self.model.fit(frame[self.FEATURES], target)
        self._fitted = True

    def predict_proba_up(self, frame: pd.DataFrame) -> float:
        if not self._fitted:
            self.fit(frame)
        if not self._fitted:
            return 0.5
        return float(self.model.predict_proba(frame[self.FEATURES].tail(1))[0, 1])
