from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from reco_trading.research.walk_forward import WalkForward


try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - dependencia opcional
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - dependencia opcional
    LGBMClassifier = None


@dataclass(slots=True)
class StackingArtifacts:
    version: str
    model_path: Path
    calibrator_path: Path | None
    metadata_path: Path


class StackingFeatureBuilder:
    """Construye matriz de features para el stack con señales cuant + ajustes meta/RL."""

    MOMENTUM_FEATURES = ["return", "ema12", "ema26", "macd", "breakout20", "volatility20", "volume_norm"]
    MEAN_REVERSION_FEATURES = ["zscore20", "rsi14", "atr14", "bb_dev"]
    MICROSTRUCTURE_FEATURES = ["obi", "cvd", "spread", "vpin", "liquidity_shock"]

    def __init__(self) -> None:
        self.feature_order: list[str] = []

    @staticmethod
    def _safe_col(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").fillna(default)
        return pd.Series(default, index=frame.index, dtype=float)

    def build(
        self,
        frame: pd.DataFrame,
        *,
        hmm_output: pd.DataFrame | dict[str, float] | None = None,
        meta_adjustments: pd.DataFrame | dict[str, float] | None = None,
        rl_adjustments: pd.DataFrame | dict[str, float] | None = None,
    ) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)

        for col in self.MOMENTUM_FEATURES + self.MEAN_REVERSION_FEATURES + self.MICROSTRUCTURE_FEATURES:
            out[col] = self._safe_col(frame, col)

        out["momentum_score"] = out[["return", "macd", "breakout20"]].mean(axis=1)
        out["mean_reversion_score"] = out[["zscore20", "rsi14", "bb_dev"]].mean(axis=1)
        out["microstructure_pressure"] = out[["obi", "vpin"]].mean(axis=1) - out["spread"]

        if hmm_output is None:
            out["hmm_state"] = 0.0
            out["hmm_confidence"] = 0.5
        elif isinstance(hmm_output, pd.DataFrame):
            out["hmm_state"] = self._safe_col(hmm_output.reindex(frame.index), "hmm_state")
            out["hmm_confidence"] = self._safe_col(hmm_output.reindex(frame.index), "hmm_confidence", 0.5)
        else:
            out["hmm_state"] = float(hmm_output.get("hmm_state", 0.0))
            out["hmm_confidence"] = float(hmm_output.get("hmm_confidence", 0.5))

        if meta_adjustments is None:
            out["meta_confidence"] = 0.5
            out["meta_weight_momentum"] = 1.0
            out["meta_weight_reversion"] = 1.0
        elif isinstance(meta_adjustments, pd.DataFrame):
            aligned = meta_adjustments.reindex(frame.index)
            out["meta_confidence"] = self._safe_col(aligned, "meta_confidence", 0.5)
            out["meta_weight_momentum"] = self._safe_col(aligned, "meta_weight_momentum", 1.0)
            out["meta_weight_reversion"] = self._safe_col(aligned, "meta_weight_reversion", 1.0)
        else:
            out["meta_confidence"] = float(meta_adjustments.get("meta_confidence", 0.5))
            out["meta_weight_momentum"] = float(meta_adjustments.get("meta_weight_momentum", 1.0))
            out["meta_weight_reversion"] = float(meta_adjustments.get("meta_weight_reversion", 1.0))

        if rl_adjustments is None:
            out["rl_size_multiplier"] = 1.0
            out["rl_threshold_shift"] = 0.0
            out["rl_risk_shift"] = 0.0
            out["rl_pause_trading"] = 0.0
        elif isinstance(rl_adjustments, pd.DataFrame):
            aligned = rl_adjustments.reindex(frame.index)
            out["rl_size_multiplier"] = self._safe_col(aligned, "rl_size_multiplier", 1.0)
            out["rl_threshold_shift"] = self._safe_col(aligned, "rl_threshold_shift", 0.0)
            out["rl_risk_shift"] = self._safe_col(aligned, "rl_risk_shift", 0.0)
            out["rl_pause_trading"] = self._safe_col(aligned, "rl_pause_trading", 0.0)
        else:
            out["rl_size_multiplier"] = float(rl_adjustments.get("size_multiplier", 1.0))
            out["rl_threshold_shift"] = float(rl_adjustments.get("threshold_shift", 0.0))
            out["rl_risk_shift"] = float(rl_adjustments.get("risk_shift", 0.0))
            out["rl_pause_trading"] = 1.0 if bool(rl_adjustments.get("pause_trading", False)) else 0.0

        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        self.feature_order = list(out.columns)
        return out


class StackingEnsemble:
    def __init__(
        self,
        *,
        model_type: str = "gradient_boosting",
        calibrator: str = "isotonic",
        model_params: dict[str, float | int] | None = None,
        artifact_dir: str | Path = "artifacts/stacking_ensemble",
    ) -> None:
        self.model_type = model_type
        self.calibrator = calibrator
        self.model_params = model_params or {}
        self.artifact_dir = Path(artifact_dir)
        self.walk_forward = WalkForward()

        self.model = self._build_base_model()
        self.calibrator_model: IsotonicRegression | LogisticRegression | None = None
        self.feature_names: list[str] = []
        self.version: str | None = None

    def _build_base_model(self):
        if self.model_type == "gradient_boosting":
            defaults = {"n_estimators": 180, "learning_rate": 0.04, "max_depth": 3, "random_state": 7}
            return GradientBoostingClassifier(**{**defaults, **self.model_params})

        if self.model_type == "xgboost":
            if XGBClassifier is None:
                raise RuntimeError("xgboost no está disponible en el entorno")
            defaults = {
                "n_estimators": 220,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
            }
            return XGBClassifier(**{**defaults, **self.model_params})

        if self.model_type == "lightgbm":
            if LGBMClassifier is None:
                raise RuntimeError("lightgbm no está disponible en el entorno")
            defaults = {"n_estimators": 250, "learning_rate": 0.05, "num_leaves": 31, "random_state": 7}
            return LGBMClassifier(**{**defaults, **self.model_params})

        raise ValueError(f"model_type inválido: {self.model_type}")

    @staticmethod
    def _normalize_timestamps(index: pd.Index, timestamps: pd.Series | None) -> pd.Series:
        if timestamps is not None:
            ts = pd.to_datetime(timestamps, utc=True)
        else:
            ts = pd.to_datetime(index, utc=True)
        return pd.Series(ts, index=index)

    def _fit_calibrator(self, oof_probs: np.ndarray, y_true: np.ndarray) -> None:
        if self.calibrator == "none":
            self.calibrator_model = None
            return

        if np.unique(y_true).size < 2:
            self.calibrator_model = None
            return

        if self.calibrator == "platt":
            cal = LogisticRegression(max_iter=1000)
            cal.fit(oof_probs.reshape(-1, 1), y_true)
            self.calibrator_model = cal
            return

        if self.calibrator == "isotonic":
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(oof_probs, y_true)
            self.calibrator_model = cal
            return

        raise ValueError(f"Calibrador inválido: {self.calibrator}")

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        timestamps: pd.Series | None = None,
        train: int = 1000,
        test: int = 200,
        step: int = 100,
        min_train: int = 400,
    ) -> None:
        aligned = features.copy()
        aligned["target"] = pd.to_numeric(target, errors="coerce").fillna(0).astype(int)
        aligned["_timestamp"] = self._normalize_timestamps(features.index, timestamps)
        aligned = aligned.sort_values("_timestamp").reset_index(drop=True)

        if len(aligned) < min_train + test:
            raise ValueError("No hay suficientes datos para walk-forward")

        self.feature_names = [c for c in aligned.columns if c not in {"target", "_timestamp"}]

        oof_probs: list[float] = []
        oof_target: list[int] = []

        has_split = False
        for train_slice, test_slice in self.walk_forward.generate_splits(
            aligned,
            train=train,
            test=test,
            step=step,
            min_train=min_train,
        ):
            has_split = True
            max_train_ts = train_slice["_timestamp"].max()
            min_test_ts = test_slice["_timestamp"].min()
            if max_train_ts >= min_test_ts:
                raise ValueError("Violación de separación temporal estricta en walk-forward")

            x_train = train_slice[self.feature_names]
            y_train = train_slice["target"]
            x_test = test_slice[self.feature_names]
            y_test = test_slice["target"]

            self.model.fit(x_train, y_train)
            probs = self.model.predict_proba(x_test)[:, 1]
            oof_probs.extend([float(v) for v in probs])
            oof_target.extend([int(v) for v in y_test.to_numpy(dtype=int)])

        if not has_split:
            raise ValueError("Walk-forward no generó ventanas válidas")

        self.model.fit(aligned[self.feature_names], aligned["target"])
        self._fit_calibrator(np.asarray(oof_probs, dtype=float), np.asarray(oof_target, dtype=int))

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if not self.feature_names:
            raise RuntimeError("El ensemble debe entrenarse antes de predecir")

        x = features[self.feature_names]
        base_probs = self.model.predict_proba(x)[:, 1]

        if self.calibrator_model is None:
            calibrated = base_probs
        elif isinstance(self.calibrator_model, IsotonicRegression):
            calibrated = self.calibrator_model.predict(base_probs)
        else:
            calibrated = self.calibrator_model.predict_proba(base_probs.reshape(-1, 1))[:, 1]

        calibrated = np.clip(calibrated, 0.0, 1.0)
        return np.column_stack([1.0 - calibrated, calibrated])

    def save_artifacts(self, version: str | None = None) -> StackingArtifacts:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        if version is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            version = f"stacking-{stamp}-{uuid4().hex[:8]}"
        self.version = version

        model_path = self.artifact_dir / f"{version}.model.joblib"
        calibrator_path = self.artifact_dir / f"{version}.calibrator.joblib"
        metadata_path = self.artifact_dir / f"{version}.metadata.json"

        joblib.dump(self.model, model_path)
        has_calibrator = self.calibrator_model is not None
        if has_calibrator:
            joblib.dump(self.calibrator_model, calibrator_path)

        metadata = {
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_type": self.model_type,
            "calibrator": self.calibrator,
            "feature_names": self.feature_names,
            "has_calibrator": has_calibrator,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return StackingArtifacts(
            version=version,
            model_path=model_path,
            calibrator_path=calibrator_path if has_calibrator else None,
            metadata_path=metadata_path,
        )

    def load_artifacts(self, version: str) -> None:
        model_path = self.artifact_dir / f"{version}.model.joblib"
        calibrator_path = self.artifact_dir / f"{version}.calibrator.joblib"
        metadata_path = self.artifact_dir / f"{version}.metadata.json"

        if not model_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"No existen artefactos para versión {version}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.model = joblib.load(model_path)
        self.feature_names = list(metadata.get("feature_names", []))
        self.model_type = str(metadata.get("model_type", self.model_type))
        self.calibrator = str(metadata.get("calibrator", self.calibrator))
        self.version = version

        if metadata.get("has_calibrator", False) and calibrator_path.exists():
            self.calibrator_model = joblib.load(calibrator_path)
        else:
            self.calibrator_model = None
