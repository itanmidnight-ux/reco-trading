from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib

from trading_system.app.ml.model_registry import ModelRegistry
from trading_system.app.models.ml_models.base import BaseModel, Probabilities
from trading_system.app.services.feature_engineering.pipeline import FeatureVector


@dataclass
class ManagedModel:
    model_name: str
    version: str
    source: str
    predictor: BaseModel


class ArtifactModel(BaseModel):
    def __init__(self, model: Any) -> None:
        self.model = model

    def predict_proba(self, f: FeatureVector) -> Probabilities:
        row = [float(getattr(f, feature_name)) for feature_name in f.__dataclass_fields__]

        if hasattr(self.model, 'predict_proba'):
            raw = self.model.predict_proba([row])
            up = float(raw[0][1])
            return Probabilities(up=up, down=1 - up)

        if hasattr(self.model, 'predict'):
            raw_pred = self.model.predict([row])
            if isinstance(raw_pred, (list, tuple)):
                first = raw_pred[0]
                up = float(first[0] if isinstance(first, (list, tuple)) else first)
            else:
                up = float(raw_pred)
            up = max(0.0, min(1.0, up))
            return Probabilities(up=up, down=1 - up)

        raise ValueError('Loaded model does not expose predict_proba or predict')


class ModelManager:
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def load_active(self, name: str) -> ManagedModel | None:
        rec = self.registry.latest(name)
        if rec is None:
            return None
        model = joblib.load(rec.path)
        return ManagedModel(model_name=name, version=rec.version, source=rec.path, predictor=ArtifactModel(model))
