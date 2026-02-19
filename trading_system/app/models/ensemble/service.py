from __future__ import annotations

from dataclasses import dataclass

from trading_system.app.ml.model_manager import ManagedModel, ModelManager
from trading_system.app.ml.model_registry import ModelRegistry
from trading_system.app.models.ml_models.base import BaseModel, Probabilities
from trading_system.app.services.feature_engineering.pipeline import FeatureVector
from trading_system.app.services.regime_detection.service import RegimeState


class SafeHoldModel(BaseModel):
    def predict_proba(self, _: FeatureVector) -> Probabilities:
        return Probabilities(up=0.5, down=0.5)


@dataclass
class EnsembleOutput:
    p_up: float
    p_down: float
    score: float
    mode: str
    model_versions: dict[str, str]
    model_sources: dict[str, str]
    regime: str


class EnsembleService:
    MODEL_REGISTRY_MAP = {
        'rf': 'rf_classifier',
        'xgb': 'xgb_classifier',
        'lstm': 'lstm_classifier',
    }

    def __init__(self, manager: ModelManager | None = None, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry()
        self.manager = manager or ModelManager(self.registry)
        self.models: dict[str, ManagedModel] = {}
        self.safe_mode = False
        self._bootstrap_models()

    def _bootstrap_models(self) -> None:
        missing_keys: list[str] = []
        for key, registry_name in self.MODEL_REGISTRY_MAP.items():
            loaded = self.manager.load_active(registry_name)
            if loaded is None:
                missing_keys.append(key)
                continue
            self.models[key] = loaded

        if missing_keys:
            self.safe_mode = True
            for key in self.MODEL_REGISTRY_MAP:
                self.models[key] = ManagedModel(
                    model_name=f'{key}_safe_hold',
                    version='safe-hold',
                    source='internal_fallback',
                    predictor=SafeHoldModel(),
                )

    def infer(self, f: FeatureVector, regime: RegimeState) -> EnsembleOutput:
        p_rf = self.models['rf'].predictor.predict_proba(f)
        p_xgb = self.models['xgb'].predictor.predict_proba(f)
        p_lstm = self.models['lstm'].predictor.predict_proba(f)
        w = regime.weights
        p_up = w['rf'] * p_rf.up + w['xgb'] * p_xgb.up + w['lstm'] * p_lstm.up
        p_down = 1 - p_up
        return EnsembleOutput(
            p_up=p_up,
            p_down=p_down,
            score=max(0.0, min(1.0, p_up)),
            mode='SAFE_HOLD' if self.safe_mode else 'ACTIVE',
            model_versions={k: v.version for k, v in self.models.items()},
            model_sources={k: v.source for k, v in self.models.items()},
            regime=regime.name,
        )
