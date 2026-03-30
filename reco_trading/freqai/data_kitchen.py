"""
Data Kitchen for FreqAI.
Handles data preprocessing and feature engineering for ML models.
"""

import logging
from typing import Any

import pandas as pd
import numpy as np

from reco_trading.constants import Config


logger = logging.getLogger(__name__)


class DataKitchen:
    """
    Data Kitchen handles data preprocessing for FreqAI.
    """
    
    def __init__(self, config: Config) -> None:
        """
        Initialize Data Kitchen.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.freqai_config = config.get("freqai", {})
        
        self.feature_parameters = self.freqai_config.get("feature_parameters", {})
        self.labeling = self.freqai_config.get("labeling", {})
        
        self.train_data = None
        self.test_data = None
        
    async def _prepare_data(
        self,
        dataframe: pd.DataFrame,
        train: bool = True,
    ) -> pd.DataFrame:
        """
        Prepare data for ML training or prediction.
        
        Args:
            dataframe: Raw OHLCV data
            train: Whether this is training data
            
        Returns:
            Processed DataFrame
        """
        df = dataframe.copy()
        
        df = self._remove_duplicates(df)
        
        df = self._fill_na(df)
        
        df = self._normalize_features(df)
        
        if train:
            df = self._add_labels(df)
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate timestamps."""
        if "date" in df.columns:
            df = df.drop_duplicates(subset=["date"], keep="first")
        return df
    
    def _fill_na(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(method="ffill")
        df[numeric_cols] = df[numeric_cols].fillna(0)
        return df
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize feature values."""
        normalize = self.feature_parameters.get("normalize", True)
        
        if not normalize:
            return df
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col == "date":
                continue
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col] = (df[col] - mean) / std
                
        return df
    
    def _add_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add labels to the dataframe.
        
        Labels can be:
        - binary: 1 if profit above threshold, 0 otherwise
        - regression: actual profit percentage
        """
        label_type = self.labeling.get("type", "binary")
        threshold = self.labeling.get("threshold", 0.02)
        forward_candles = self.labeling.get("forward_candles", 1)
        
        if label_type == "binary":
            future_returns = df["close"].shift(-forward_candles) / df["close"] - 1
            df["label"] = (future_returns > threshold).astype(int)
            
        elif label_type == "regression":
            df["label"] = df["close"].shift(-forward_candles) / df["close"] - 1
            
        elif label_type == "triangular":
            df["label"] = self._create_triangular_labels(df, threshold)
        
        df = df.dropna(subset=["label"])
        
        return df
    
    def _create_triangular_labels(
        self,
        df: pd.DataFrame,
        threshold: float,
    ) -> pd.Series:
        """
        Create triangular labels (-1, 0, 1).
        
        Args:
            df: DataFrame with prices
            threshold: Threshold for classification
            
        Returns:
            Labels series
        """
        future_returns = df["close"].shift(-1) / df["close"] - 1
        
        labels = pd.Series(0, index=df.index)
        labels[future_returns > threshold] = 1
        labels[future_returns < -threshold] = -1
        
        return labels
    
    def split_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets.
        
        Args:
            df: DataFrame to split
            test_size: Fraction for test set
            
        Returns:
            Tuple of (train, test) DataFrames
        """
        split_idx = int(len(df) * (1 - test_size))
        
        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()
        
        self.train_data = train
        self.test_data = test
        
        return train, test
    
    def get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with features
        """
        features = df.copy()
        
        exclude_cols = ["date", "label", "symbol"]
        feature_cols = [c for c in features.columns if c not in exclude_cols]
        
        return features[feature_cols]
    
    def get_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Extract labels from dataframe.
        
        Args:
            df: DataFrame with labels
            
        Returns:
            Labels series
        """
        if "label" not in df.columns:
            raise ValueError("No labels found in dataframe")
            
        return df["label"]
    
    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        std_threshold: float = 3.0,
    ) -> pd.DataFrame:
        """
        Remove outliers using z-score.
        
        Args:
            df: DataFrame
            columns: Columns to check
            std_threshold: Z-score threshold
            
        Returns:
            DataFrame without outliers
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
        df_clean = df.copy()
        
        for col in columns:
            if col in ["date", "label"]:
                continue
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            if std > 0:
                z_scores = np.abs((df_clean[col] - mean) / std)
                df_clean = df_clean[z_scores < std_threshold]
                
        return df_clean
    
    def feature_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        max_features: int = 50,
    ) -> list[str]:
        """
        Select top features based on correlation.
        
        Args:
            X: Features
            y: Labels
            max_features: Maximum number of features
            
        Returns:
            List of selected feature names
        """
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        
        top_features = correlations.head(max_features).index.tolist()
        
        return top_features
