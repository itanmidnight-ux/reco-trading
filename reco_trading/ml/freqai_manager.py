from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FreqAIConfig:
    """Configuration for FreqAI - ML-based trading."""
    enabled: bool = True
    model_type: str = "lightgbm"
    model_classifier: bool = True
    train_period: int = 30
    prediction_threshold: float = 0.55
    use_learn_threshold: bool = True
    identifier: str = "freqai"
    feature_parameters: dict = field(default_factory=dict)
    data_split_parameters: dict = field(default_factory=dict)
    model_training_parameters: dict = field(default_factory=dict)
    auto_retrain: bool = True
    retrain_interval_hours: int = 6


@dataclass
class ModelInfo:
    """Information about a trained model."""
    model_id: str
    pair: str
    created_at: datetime
    accuracy: float = 0.0
    samples: int = 0
    features: list[str] = field(default_factory=list)
    path: str = ""


class FreqAIManager:
    """
    FreqAI - Enhanced Machine Learning module for trading.
    Supports multiple model types and auto-training.
    """

    def __init__(self, config: FreqAIConfig | None = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or FreqAIConfig()
        self.models: dict[str, ModelInfo] = {}
        self.is_training: bool = False
        self._training_task: asyncio.Task | None = None
        self._data_cache: dict[str, pd.DataFrame] = {}
        
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the ML model."""
        
        if not self.config.enabled:
            self.logger.info("FreqAI is disabled")
            return
        
        try:
            if self.config.model_type == "lightgbm":
                import lightgbm as lgb
                self.model_class = lgb.LGBMClassifier if self.config.model_classifier else lgb.LGBMRegressor
                self.logger.info("Initialized LightGBM model")
            elif self.config.model_type == "xgboost":
                import xgboost as xgb
                self.model_class = xgb.XGBClassifier if self.config.model_classifier else xgb.XGBRegressor
                self.logger.info("Initialized XGBoost model")
            elif self.config.model_type == "random_forest":
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                self.model_class = RandomForestClassifier if self.config.model_classifier else RandomForestRegressor
                self.logger.info("Initialized Random Forest model")
            else:
                self.logger.warning(f"Unknown model type: {self.config.model_type}, using default")
                self.model_class = None
                
        except ImportError as exc:
            self.logger.warning(f"ML library not available: {exc}")
            self.model_class = None

    async def start(self) -> None:
        """Start the FreqAI training scheduler."""
        
        if not self.config.enabled:
            return
        
        if self.config.auto_retrain:
            self._training_task = asyncio.create_task(self._training_loop())
            self.logger.info("FreqAI auto-training scheduler started")

    async def stop(self) -> None:
        """Stop FreqAI."""
        
        if self._training_task:
            self._training_task.cancel()
            try:
                await self._training_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("FreqAI stopped")

    async def _training_loop(self) -> None:
        """Background training loop."""
        
        while True:
            try:
                await asyncio.sleep(3600 * self.config.retrain_interval_hours)
                
                if not self.is_training:
                    self.logger.info("Starting scheduled FreqAI model training")
                    
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.logger.error(f"Training loop error: {exc}")

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML training."""
        
        if df.empty:
            return df
        
        features = df.copy()
        
        for col in ["open", "high", "low", "close", "volume"]:
            if col in features.columns:
                features[f"{col}_returns"] = features[col].pct_change()
                features[f"{col}_sma_5"] = features[col].rolling(5).mean()
                features[f"{col}_sma_20"] = features[col].rolling(20).mean()
                features[f"{col}_std_20"] = features[col].rolling(20).std()
        
        if "close" in features.columns:
            features["rsi"] = self._calculate_rsi(features["close"])
            features["macd"] = self._calculate_macd(features["close"])
            features["bb_position"] = self._calculate_bollinger_position(features["close"])
        
        features = features.dropna()
        
        return features

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD indicator."""
        
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=9, adjust=False).mean()
        
        return macd - signal

    def _calculate_bollinger_position(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Bollinger Bands position."""
        
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        
        position = (prices - lower) / (upper - lower + 1e-10)
        
        return position

    async def train(
        self,
        df: pd.DataFrame,
        target_column: str = "close",
        pair: str = "UNKNOWN",
    ) -> bool:
        """Train the model on historical data."""
        
        if not self.config.enabled or self.model_class is None:
            return False
        
        if df.empty or len(df) < 100:
            self.logger.warning(f"Not enough data to train for {pair}")
            return False
        
        try:
            self.is_training = True
            
            features_df = self.prepare_features(df)
            
            feature_cols = [c for c in features_df.columns if c != target_column]
            
            X = features_df[feature_cols].values
            y = (features_df[target_column].shift(-1) > features_df[target_column]).astype(int).values
            
            X = X[:-1]
            y = y[:-1]
            
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            model = self.model_class(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
            )
            
            model.fit(X_train, y_train)
            
            # Store the trained model in memory for predictions
            setattr(self, f"_trained_model_{pair}", model)
            
            accuracy = model.score(X_test, y_test)
            
            model_id = f"{pair}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.models[pair] = ModelInfo(
                model_id=model_id,
                pair=pair,
                created_at=datetime.now(timezone.utc),
                accuracy=accuracy,
                samples=len(X_train),
                features=feature_cols,
            )
            
            self.logger.info(
                f"Model trained for {pair}: accuracy={accuracy:.2%}, samples={len(X_train)}"
            )
            
            self.is_training = False
            return True
            
        except Exception as exc:
            self.logger.error(f"Training failed for {pair}: {exc}")
            self.is_training = False
            return False

    async def predict(self, df: pd.DataFrame, pair: str = "UNKNOWN") -> dict | None:
        """Make predictions using trained model."""
        
        if not self.config.enabled or self.model_class is None:
            return None
        
        if pair not in self.models:
            return None
        
        try:
            features_df = self.prepare_features(df)
            
            if features_df.empty:
                return None
            
            feature_cols = self.models[pair].features
            X = features_df[feature_cols].iloc[-1:].values
            
            # Use the TRAINED model stored in memory, NOT create a new one
            trained_model = getattr(self, f"_trained_model_{pair}", None)
            if trained_model is None:
                self.logger.warning(f"No trained model in memory for {pair}, prediction unavailable")
                return None
            
            # Get prediction from the actual trained model
            if hasattr(trained_model, "predict_proba"):
                probabilities = trained_model.predict_proba(X)
                probability = float(probabilities[0][1]) if probabilities.shape[1] > 1 else 0.5
            elif hasattr(trained_model, "predict"):
                prediction = trained_model.predict(X)
                probability = float(prediction[0])
            else:
                self.logger.warning(f"Model for {pair} has no predict method")
                return None
            
            threshold = self.config.prediction_threshold
            
            if self.config.use_learn_threshold and self.models[pair].accuracy > 0:
                threshold = max(0.5, min(0.7, self.models[pair].accuracy))
            
            direction = "BUY" if probability >= threshold else "SELL" if probability <= (1 - threshold) else "HOLD"
            
            return {
                "direction": direction,
                "confidence": probability,
                "threshold": threshold,
                "model_accuracy": self.models[pair].accuracy,
            }
            
        except Exception as exc:
            self.logger.error(f"Prediction failed for {pair}: {exc}")
            return None

    def get_model_info(self, pair: str) -> ModelInfo | None:
        """Get model information for a pair."""
        return self.models.get(pair)

    def get_all_models(self) -> dict[str, ModelInfo]:
        """Get all trained models."""
        return self.models.copy()

    def get_stats(self) -> dict:
        """Get FreqAI statistics."""
        
        accuracies = [m.accuracy for m in self.models.values()]
        
        return {
            "enabled": self.config.enabled,
            "model_type": self.config.model_type,
            "is_training": self.is_training,
            "total_models": len(self.models),
            "avg_accuracy": np.mean(accuracies) if accuracies else 0,
            "auto_retrain": self.config.auto_retrain,
        }

    def clear_models(self) -> None:
        """Clear all trained models."""
        self.models.clear()
        self.logger.info("All FreqAI models cleared")


def create_freqai_config(**kwargs) -> FreqAIConfig:
    """Factory function to create FreqAI config."""
    return FreqAIConfig(**kwargs)
