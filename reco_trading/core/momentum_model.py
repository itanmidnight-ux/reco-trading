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
        self.model.fit(frame[self.FEATURES], frame['target_up'])
        self._fitted = True

    def predict_proba_up(self, frame: pd.DataFrame) -> float:
        if not self._fitted:
            self.fit(frame)
        return float(self.model.predict_proba(frame[self.FEATURES].tail(1))[0, 1])
