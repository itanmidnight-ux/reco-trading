"""
Feature Engineering for FreqAI.
Automatic feature generation for ML models.
"""

import logging
from typing import Any

import pandas as pd
import numpy as np

from reco_trading.constants import Config


logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Automatic feature engineering for FreqAI.
    Creates technical indicators and statistical features.
    """
    
    def __init__(self, config: Config) -> None:
        """
        Initialize Feature Engineer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.freqai_config = config.get("freqai", {})
        
        self.include_shifted_candles = self.freqai_config.get(
            "include_shifted_candles", True
        )
        self.include_volume = self.freqai_config.get("include_volume", True)
        
    def engineer_features(
        self,
        dataframe: pd.DataFrame,
        pair: str = "",
    ) -> pd.DataFrame:
        """
        Engineer features from OHLCV data.
        
        Args:
            dataframe: DataFrame with OHLCV data
            pair: Trading pair name
            
        Returns:
            DataFrame with engineered features
        """
        df = dataframe.copy()
        
        df = self._add_price_features(df)
        
        df = self._add_rolling_features(df)
        
        df = self._add_momentum_features(df)
        
        df = self._add_volatility_features(df)
        
        if self.include_volume:
            df = self._add_volume_features(df)
            
        if self.include_shifted_candles:
            df = self._add_shifted_features(df)
            
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price features."""
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        
        df["high_low_ratio"] = df["high"] / df["low"]
        df["close_open_ratio"] = df["close"] / df["open"]
        
        df["price_range"] = df["high"] - df["low"]
        df["price_range_pct"] = (df["high"] - df["low"]) / df["close"]
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window features."""
        windows = [5, 10, 20, 50]
        
        for window in windows:
            df[f"sma_{window}"] = df["close"].rolling(window).mean()
            df[f"ema_{window}"] = df["close"].ewm(span=window).mean()
            df[f"std_{window}"] = df["close"].rolling(window).std()
            
            df[f"volume_sma_{window}"] = df["volume"].rolling(window).mean()
            
        for window in [10, 20]:
            df[f"returns_mean_{window}"] = df["returns"].rolling(window).mean()
            df[f"returns_std_{window}"] = df["returns"].rolling(window).std()
            
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        for period in [7, 14, 21]:
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            
            rs = gain / loss.replace(0, np.nan)
            df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
        
        for short, long in [(12, 26), (8, 21)]:
            ema_short = df["close"].ewm(span=short).mean()
            ema_long = df["close"].ewm(span=long).mean()
            df[f"macd_{short}_{long}"] = ema_short - ema_long
            
        df["macd_signal"] = df["macd_12_26"].ewm(span=9).mean()
        df["macd_hist"] = df["macd_12_26"] - df["macd_signal"]
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features."""
        for period in [10, 20, 50]:
            df[f"bb_upper_{period}"] = df["close"].rolling(period).mean() + (
                df["close"].rolling(period).std() * 2
            )
            df[f"bb_lower_{period}"] = df["close"].rolling(period).mean() - (
                df["close"].rolling(period).std() * 2
            )
            df[f"bb_width_{period}"] = (
                df[f"bb_upper_{period}"] - df[f"bb_lower_{period}"]
            ) / df["close"]
            
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        df["volume_price"] = df["volume"] * df["close"]
        
        df["volume_change"] = df["volume"].pct_change()
        
        for period in [10, 20]:
            df[f"volume_sma_{period}"] = df["volume"].rolling(period).mean()
            df[f"volume_ratio_{period}"] = df["volume"] / df[f"volume_sma_{period}"]
            
        df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
        
        return df
    
    def _add_shifted_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add shifted (lagged) features."""
        shifts = [1, 2, 3, 5, 10]
        
        for shift in shifts:
            df[f"close_lag_{shift}"] = df["close"].shift(shift)
            df[f"returns_lag_{shift}"] = df["returns"].shift(shift)
            df[f"volume_lag_{shift}"] = df["volume"].shift(shift)
            
        return df
    
    def get_feature_names(self, df: pd.DataFrame) -> list[str]:
        """
        Get list of feature names.
        
        Args:
            df: DataFrame with features
            
        Returns:
            List of feature column names
        """
        exclude = ["date", "open", "high", "low", "close", "volume", "label"]
        features = [c for c in df.columns if c not in exclude]
        
        return features


class FeatureSelector:
    """
    Select best features for ML models.
    """
    
    def __init__(self, config: Config) -> None:
        """Initialize Feature Selector."""
        self.config = config
        
    def select_top_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        max_features: int = 50,
    ) -> list[str]:
        """
        Select top features based on correlation.
        
        Args:
            X: Features DataFrame
            y: Labels
            max_features: Maximum number of features
            
        Returns:
            List of selected feature names
        """
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        
        valid_corrs = correlations.dropna()
        
        top_features = valid_corrs.head(max_features).index.tolist()
        
        logger.info(f"Selected {len(top_features)} top features")
        
        return top_features
    
    def remove_correlated_features(
        self,
        X: pd.DataFrame,
        threshold: float = 0.95,
    ) -> pd.DataFrame:
        """
        Remove highly correlated features.
        
        Args:
            X: Features DataFrame
            threshold: Correlation threshold
            
        Returns:
            DataFrame with removed features
        """
        corr_matrix = X.corr().abs()
        
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [
            column for column in upper.columns 
            if any(upper[column] > threshold)
        ]
        
        logger.info(f"Dropping {len(to_drop)} highly correlated features")
        
        return X.drop(columns=to_drop)


__all__ = ["FeatureEngineer", "FeatureSelector"]
