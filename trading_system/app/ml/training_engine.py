from __future__ import annotations

import pandas as pd

from trading_system.app.ml.model_registry import ModelRegistry
from trading_system.app.training_pipeline.pipeline import EnsembleTrainingPipeline


class TrainingEngine:
    def __init__(self, model_dir: str) -> None:
        self.pipeline = EnsembleTrainingPipeline(model_dir)
        self.registry = ModelRegistry()

    def train(self, df: pd.DataFrame, feature_cols: list[str], target_col: str = 'target') -> dict:
        result = self.pipeline.run(df, feature_cols, target_col=target_col)
        self.registry.register('rf_classifier', result.model_version, result.model_path)
        return {
            'model_version': result.model_version,
            'mean_cv_accuracy': result.mean_cv_accuracy,
            'feature_importance': result.feature_importance,
        }
