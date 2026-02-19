from __future__ import annotations

from trading_system.app.models.ml_models.base import BaseModel, Probabilities
from trading_system.app.services.feature_engineering.pipeline import FeatureVector


class RandomForestModel(BaseModel):
    def predict_proba(self, f: FeatureVector) -> Probabilities:
        up = min(0.99, max(0.01, 0.5 + 0.2 * (f.ema9 > f.ema21) - 0.15 * (f.rsi > 70)))
        return Probabilities(up=up, down=1 - up)


class XGBoostModel(BaseModel):
    def predict_proba(self, f: FeatureVector) -> Probabilities:
        up = min(0.99, max(0.01, 0.5 + 0.25 * f.orderbook_imbalance + 0.1 * f.breakout_score))
        return Probabilities(up=up, down=1 - up)


class LSTMModel(BaseModel):
    def predict_proba(self, f: FeatureVector) -> Probabilities:
        momentum = 0.15 if f.zscore_price > 0 else -0.15
        up = min(0.99, max(0.01, 0.5 + momentum - 0.1 * (f.volatility > 1.5)))
        return Probabilities(up=up, down=1 - up)
