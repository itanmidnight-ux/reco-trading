from __future__ import annotations

import pandas as pd
from xgboost import XGBClassifier


class MeanReversionModel:
    FEATURES = ['zscore20', 'rsi14', 'atr14', 'bb_dev']

    def __init__(self) -> None:
        self.model = XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective='binary:logistic',
            eval_metric='logloss',
        )
        self._fitted = False

    def fit(self, frame: pd.DataFrame) -> None:
        target = frame['target_reversion']
        if target.nunique(dropna=True) < 2:
            # Fallback defensivo para evitar error cuando solo hay una clase.
            self._fitted = False
            return
        self.model.fit(frame[self.FEATURES], target)
        self._fitted = True

    def predict_reversion(self, frame: pd.DataFrame) -> float:
        if not self._fitted:
            self.fit(frame)
        if not self._fitted:
            return 0.5
        return float(self.model.predict_proba(frame[self.FEATURES].tail(1))[0, 1])
