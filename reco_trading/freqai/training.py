from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reco_trading.constants import Config
from reco_trading.freqai.feature_engineering import FeatureEngineer
from reco_trading.freqai.prediction_models import (
    LightGBMModel,
    XGBoostModel,
    RandomForestModel,
    HAS_LIGHTGBM,
    HAS_XGBOOST,
    HAS_SKLEARN,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    model_type: str = "LightGBMClassifier"
    train_period_days: int = 14
    retrain_interval_hours: int = 6
    label_type: str = "binary"
    label_threshold: float = 0.02
    forward_candles: int = 1
    test_size: float = 0.2
    max_features: int = 50
    save_path: str = "models"


@dataclass
class TrainingStats:
    training_samples: int = 0
    test_samples: int = 0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    last_train_time: datetime | None = None
    next_train_time: datetime | None = None


class FreqAITrainer:
    """Manages model training and retraining for FreqAI."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.freqai_config = config.get("freqai", {})
        
        self.training_config = TrainingConfig(
            model_type=self.freqai_config.get("model", "LightGBMClassifier"),
            train_period_days=self.freqai_config.get("train_period_days", 14),
            retrain_interval_hours=self.freqai_config.get("retrain_interval_hours", 6),
            label_type=self.freqai_config.get("label_type", "binary"),
            label_threshold=self.freqai_config.get("label_threshold", 0.02),
            forward_candles=self.freqai_config.get("forward_candles", 1),
            test_size=self.freqai_config.get("test_size_size", 0.2),
            max_features=self.freqai_config.get("max_features", 50),
            save_path=self.freqai_config.get("save_path", "models"),
        )
        
        self.feature_engineer = FeatureEngineer(config)
        self.model: LightGBMModel | XGBoostModel | RandomForestModel | None = None
        self.stats = TrainingStats()
        self._is_training = False
        self._last_data_timestamp: datetime | None = None
        
    def _create_model(self) -> LightGBMModel | XGBoostModel | RandomForestModel:
        """Create model instance based on configuration."""
        if self.training_config.model_type == "LightGBMClassifier":
            if not HAS_LIGHTGBM:
                logger.warning("LightGBM not available, using RandomForest fallback")
                return RandomForestModel(self.config)
            return LightGBMModel(self.config)
        elif self.training_config.model_type == "XGBoostClassifier":
            if not HAS_XGBOOST:
                logger.warning("XGBoost not available, using RandomForest fallback")
                return RandomForestModel(self.config)
            return XGBoostModel(self.config)
        else:
            return RandomForestModel(self.config)

    def _create_labels(self, df: pd.DataFrame) -> pd.Series:
        """Create labels for training."""
        forward = self.training_config.forward_candles
        threshold = self.training_config.label_threshold
        
        future_price = df["close"].shift(-forward)
        returns = (future_price / df["close"]) - 1
        
        if self.training_config.label_type == "binary":
            labels = (returns > threshold).astype(int)
        elif self.training_config.label_type == "triangular":
            labels = pd.Series(0, index=df.index)
            labels[returns > threshold] = 1
            labels[returns < -threshold] = -1
        else:
            labels = returns
            
        return labels

    async def train(self, data: pd.DataFrame) -> bool:
        """Train the model on provided data."""
        if self._is_training:
            logger.warning("Training already in progress")
            return False
            
        self._is_training = True
        try:
            logger.info(f"Training model on {len(data)} samples...")
            
            df = self.feature_engineer.engineer_features(data)
            labels = self._create_labels(df)
            
            valid_idx = labels.notna()
            df = df[valid_idx]
            labels = labels[valid_idx]
            
            if len(df) < 100:
                logger.error("Insufficient training data")
                return False
                
            feature_names = self.feature_engineer.get_feature_names(df)
            X = df[feature_names].fillna(0)
            y = labels
            
            self.model = self._create_model()
            self.model.train(X, y)
            
            self.stats.training_samples = len(X)
            self.stats.last_train_time = datetime.now(timezone.utc)
            self.stats.next_train_time = datetime.now(timezone.utc) + timedelta(
                hours=self.training_config.retrain_interval_hours
            )
            
            await self._evaluate(X, y)
            
            await self._save_model()
            
            logger.info(f"Training completed. Samples: {self.stats.training_samples}")
            return True
            
        except Exception as e:
            logger.exception(f"Training failed: {e}")
            return False
        finally:
            self._is_training = False

    async def _evaluate(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Evaluate model on test set."""
        if not self.model or not HAS_SKLEARN:
            return
            
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.training_config.test_size, random_state=42
            )
            
            self.stats.test_samples = len(X_test)
            
            y_pred = self.model.model.predict(X_test)
            
            self.stats.accuracy = accuracy_score(y_test, y_pred)
            self.stats.precision = precision_score(y_test, y_pred, zero_division=0)
            self.stats.recall = recall_score(y_test, y_pred, zero_division=0)
            self.stats.f1 = f1_score(y_test, y_pred, zero_division=0)
            
            logger.info(
                f"Model evaluation - Accuracy: {self.stats.accuracy:.3f}, "
                f"Precision: {self.stats.precision:.3f}, F1: {self.stats.f1:.3f}"
            )
            
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")

    async def _save_model(self) -> None:
        """Save trained model to disk."""
        if not self.model:
            return
            
        try:
            save_path = Path(self.training_config.save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            model_file = save_path / "freqai_model.pkl"
            self.model.save(str(model_file))
            logger.info(f"Model saved to {model_file}")
            
        except Exception as e:
            logger.warning(f"Model save failed: {e}")

    async def load_model(self) -> bool:
        """Load pre-trained model from disk."""
        try:
            model_file = Path(self.training_config.save_path) / "freqai_model.pkl"
            if not model_file.exists():
                logger.info("No pre-trained model found")
                return False
                
            self.model = self._create_model()
            self.model.load(str(model_file))
            logger.info(f"Model loaded from {model_file}")
            return True
            
        except Exception as e:
            logger.warning(f"Model load failed: {e}")
            return False

    def should_retrain(self) -> bool:
        """Check if model should be retrained."""
        if not self.stats.last_train_time:
            return True
            
        if not self.stats.next_train_time:
            return True
            
        return datetime.now(timezone.utc) >= self.stats.next_train_time

    def get_stats(self) -> dict[str, Any]:
        """Get training statistics."""
        return {
            "training_samples": self.stats.training_samples,
            "test_samples": self.stats.test_samples,
            "accuracy": self.stats.accuracy,
            "precision": self.stats.precision,
            "recall": self.stats.recall,
            "f1": self.stats.f1,
            "last_train_time": self.stats.last_train_time.isoformat() if self.stats.last_train_time else None,
            "next_train_time": self.stats.next_train_time.isoformat() if self.stats.next_train_time else None,
            "is_training": self._is_training,
        }


class FreqAIPredictor:
    """Handles real-time predictions using trained FreqAI models."""

    def __init__(self, trainer: FreqAITrainer) -> None:
        self.trainer = trainer
        self.feature_engineer = trainer.feature_engineer
        self._prediction_cache: dict[str, tuple[pd.Series, datetime]] = {}

    async def predict(self, data: pd.DataFrame) -> dict[str, Any]:
        """Generate prediction from current market data."""
        if not self.trainer.model or not self.trainer.model.is_trained:
            return {"prediction": "NO_MODEL", "confidence": 0.0, "action": "HOLD"}

        try:
            df = self.feature_engineer.engineer_features(data)
            feature_names = self.feature_engineer.get_feature_names(df)
            X = df[feature_names].fillna(0).iloc[-1:]

            prediction = self.trainer.model.predict(X).iloc[0]
            
            proba = None
            if hasattr(self.trainer.model, "predict_proba"):
                proba = self.trainer.model.predict_proba(X)
            
            confidence = 0.5
            if proba is not None and len(proba) > 0:
                confidence = float(proba.iloc[0].max())

            if prediction == 1:
                action = "BUY"
            elif prediction == -1:
                action = "SELL"
            else:
                action = "HOLD"

            return {
                "prediction": int(prediction),
                "confidence": confidence,
                "action": action,
                "probabilities": proba.to_dict() if proba is not None else None,
            }

        except Exception as e:
            logger.exception(f"Prediction failed: {e}")
            return {"prediction": "ERROR", "confidence": 0.0, "action": "HOLD", "error": str(e)}
