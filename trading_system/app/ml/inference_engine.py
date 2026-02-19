from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MLPrediction:
    p_win: float
    model_name: str
    model_version: str


class InferenceEngine:
    def infer(self, model, feature_vector: list[float], model_name: str, model_version: str) -> MLPrediction:
        if model is None:
            return MLPrediction(p_win=0.5, model_name=model_name, model_version='none')
        proba = model.predict_proba([feature_vector])[0][1]
        return MLPrediction(p_win=float(proba), model_name=model_name, model_version=model_version)
