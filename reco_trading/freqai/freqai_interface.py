"""
FreqAI Interface for Reco-Trading.
Base classes for machine learning integration.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from reco_trading.constants import Config


logger = logging.getLogger(__name__)


class IFreqaiModel(ABC):
    """
    Base class for FreqAI models.
    All ML models must inherit from this class.
    """
    
    def __init__(self, config: Config) -> None:
        """
        Initialize the model.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.is_trained = False
        
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Make predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the model.
        
        Args:
            path: Path to save model
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the model.
        
        Args:
            path: Path to load model from
        """
        pass
    
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame | None:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict
            
        Returns:
            DataFrame with probabilities or None
        """
        return None


class FreqAI:
    """
    Main FreqAI controller.
    Manages ML models and data processing.
    """
    
    def __init__(self, config: Config) -> None:
        """
        Initialize FreqAI.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model: IFreqaiModel | None = None
        self.data_kitchen = None
        self.is_initialized = False
        
        self.freqai_config = config.get("freqai", {})
        self.model_type = self.freqai_config.get("model", "LightGBMClassifier")
        self.train_period = self.freqai_config.get("train_period_days", 14)
        self.label_type = self.freqai_config.get("label_type", "binary")
        
    async def start(self) -> None:
        """Start FreqAI and initialize models."""
        logger.info("Starting FreqAI...")
        
        await self._initialize_data_kitchen()
        await self._initialize_model()
        
        self.is_initialized = True
        logger.info("FreqAI initialized successfully")
        
    async def _initialize_data_kitchen(self) -> None:
        """Initialize the data kitchen."""
        from reco_trading.freqai.data_kitchen import DataKitchen
        
        self.data_kitchen = DataKitchen(self.config)
        
    async def _initialize_model(self) -> None:
        """Initialize the ML model."""
        if self.model_type == "LightGBMClassifier":
            from reco_trading.freqai.prediction_models import LightGBMModel
            self.model = LightGBMModel(self.config)
        elif self.model_type == "XGBoostClassifier":
            from reco_trading.freqai.prediction_models import XGBoostModel
            self.model = XGBoostModel(self.config)
        elif self.model_type == "RandomForest":
            from reco_trading.freqai.prediction_models import RandomForestModel
            self.model = RandomForestModel(self.config)
        else:
            logger.warning(f"Unknown model type: {self.model_type}, using LightGBM")
            from reco_trading.freqai.prediction_models import LightGBMModel
            self.model = LightGBMModel(self.config)
    
    async def train(self, data: pd.DataFrame) -> None:
        """
        Train the model on provided data.
        
        Args:
            data: Training data with features and labels
        """
        if not self.model:
            raise ValueError("Model not initialized")
        
        logger.info("Training FreqAI model...")
        
        X = data.drop("label", axis=1, errors="ignore")
        y = data["label"] if "label" in data.columns else None
        
        if y is None:
            raise ValueError("No labels found in training data")
        
        self.model.train(X, y)
        logger.info("Model training completed")
        
    async def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Make predictions on data.
        
        Args:
            data: Features to predict
            
        Returns:
            Predictions
        """
        if not self.model:
            raise ValueError("Model not initialized")
        
        return self.model.predict(data)
    
    async def shutdown(self) -> None:
        """Shutdown FreqAI and cleanup resources."""
        logger.info("Shutting down FreqAI...")
        self.is_initialized = False


class FreqaiWrapper:
    """
    Wrapper to integrate FreqAI with strategies.
    """
    
    def __init__(self, config: Config) -> None:
        """
        Initialize FreqAI wrapper.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.freqai = None
        
    async def start(self) -> None:
        """Start FreqAI."""
        if self.config.get("freqai", {}).get("enabled", False):
            self.freqai = FreqAI(self.config)
            await self.freqai.start()
            
    async def predict(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Add ML predictions to dataframe.
        
        Args:
            dataframe: DataFrame with features
            
        Returns:
            DataFrame with added predictions
        """
        if not self.freqai or not self.freqai.is_initialized:
            return dataframe
        
        predictions = await self.freqai.predict(dataframe)
        dataframe["ai_prediction"] = predictions
        
        return dataframe
    
    async def shutdown(self) -> None:
        """Shutdown FreqAI."""
        if self.freqai:
            await self.freqai.shutdown()
