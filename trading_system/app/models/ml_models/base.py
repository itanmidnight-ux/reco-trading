from __future__ import annotations

from dataclasses import dataclass

from trading_system.app.services.feature_engineering.pipeline import FeatureVector


@dataclass
class Probabilities:
    up: float
    down: float


class BaseModel:
    def predict_proba(self, _: FeatureVector) -> Probabilities:
        raise NotImplementedError
