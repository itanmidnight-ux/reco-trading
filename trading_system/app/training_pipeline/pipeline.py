from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class TrainingResult:
    model_version: str
    mean_cv_accuracy: float
    feature_importance: dict[str, float]
    model_path: str


class EnsembleTrainingPipeline:
    def __init__(self, model_dir: str) -> None:
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def run(self, df: pd.DataFrame, feature_cols: list[str], target_col: str = 'target') -> TrainingResult:
        df = df.sort_values('ts').reset_index(drop=True)
        X = df[feature_cols]
        y = df[target_col]

        splitter = TimeSeriesSplit(n_splits=4)
        scores: list[float] = []
        model = RandomForestClassifier(n_estimators=200, random_state=42)

        for train_idx, test_idx in splitter.split(X):
            x_train, x_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            scores.append(accuracy_score(y_test, pred))

        model.fit(X, y)
        version = f'rf_v{len(list(self.model_dir.glob("rf_*.joblib"))) + 1}'
        model_path = self.model_dir / f'{version}.joblib'
        joblib.dump(model, model_path)

        importances = {col: float(imp) for col, imp in zip(feature_cols, model.feature_importances_)}
        return TrainingResult(
            model_version=version,
            mean_cv_accuracy=float(sum(scores) / len(scores)) if scores else 0.0,
            feature_importance=importances,
            model_path=str(model_path),
        )
