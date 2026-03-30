"""
Prediction Models for FreqAI.
ML models for price prediction.
"""

import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from reco_trading.constants import Config


logger = logging.getLogger(__name__)


try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not available")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not available")

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("Scikit-learn not available")


class BaseModel:
    """Base class for all prediction models."""
    
    def __init__(self, config: Config) -> None:
        self.config = config
        self.model = None
        self.is_trained = False
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the model."""
        raise NotImplementedError
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions."""
        raise NotImplementedError
        
    def save(self, path: str) -> None:
        """Save model to file."""
        if self.model:
            with open(path, "wb") as f:
                pickle.dump(self.model, f)
            logger.info(f"Model saved to {path}")
        
    def load(self, path: str) -> None:
        """Load model from file."""
        if Path(path).exists():
            with open(path, "rb") as f:
                self.model = pickle.load(f)
            self.is_trained = True
            logger.info(f"Model loaded from {path}")


class LightGBMModel(BaseModel):
    """LightGBM classifier/regressor for price prediction."""
    
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.freqai_config = config.get("freqai", {})
        
        self.params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_estimators": 100,
        }
        
        if not HAS_LIGHTGBM:
            logger.warning("LightGBM not installed, using fallback")
            self.use_fallback = True
        else:
            self.use_fallback = False
            
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train LightGBM model."""
        if self.use_fallback:
            self._train_fallback(X_train, y_train)
            return
            
        X = X_train.fillna(0)
        
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X, y_train)
        
        self.is_trained = True
        logger.info("LightGBM model trained successfully")
        
    def _train_fallback(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fallback training using simple statistics."""
        if not HAS_SKLEARN:
            logger.error("Need sklearn or lightgbm for training")
            return
            
        X = X_train.fillna(0)
        
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(X, y_train)
        
        self.is_trained = True
        logger.info("Fallback model trained successfully")
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained")
            
        X = X.fillna(0)
        
        if hasattr(self.model, "predict"):
            predictions = self.model.predict(X)
            
        return pd.Series(predictions, index=X.index)
        
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame | None:
        """Predict class probabilities."""
        if not self.is_trained:
            return None
            
        X = X.fillna(0)
        
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            return pd.DataFrame(proba, index=X.index)
            
        return None


class XGBoostModel(BaseModel):
    """XGBoost classifier/regressor for price prediction."""
    
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        
        self.params = {
            "objective": "binary:logistic",
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
        
        if not HAS_XGBOOST:
            logger.warning("XGBoost not installed, using fallback")
            self.use_fallback = True
        else:
            self.use_fallback = False
            
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train XGBoost model."""
        if self.use_fallback:
            self._train_fallback(X_train, y_train)
            return
            
        X = X_train.fillna(0)
        
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, y_train)
        
        self.is_trained = True
        logger.info("XGBoost model trained successfully")
        
    def _train_fallback(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fallback training using sklearn."""
        if not HAS_SKLEARN:
            logger.error("Need sklearn or xgboost for training")
            return
            
        X = X_train.fillna(0)
        
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(X, y_train)
        
        self.is_trained = True
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained")
            
        X = X.fillna(0)
        
        predictions = self.model.predict(X)
            
        return pd.Series(predictions, index=X.index)


class RandomForestModel(BaseModel):
    """Random Forest classifier/regressor."""
    
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required for RandomForest")
            
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train Random Forest model."""
        X = X_train.fillna(0)
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X, y_train)
        
        self.is_trained = True
        logger.info("Random Forest model trained successfully")
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained")
            
        X = X.fillna(0)
        
        predictions = self.model.predict(X)
            
        return pd.Series(predictions, index=X.index)


__all__ = [
    "LightGBMModel",
    "XGBoostModel", 
    "RandomForestModel",
    "HAS_LIGHTGBM",
    "HAS_XGBOOST",
    "HAS_SKLEARN",
]
