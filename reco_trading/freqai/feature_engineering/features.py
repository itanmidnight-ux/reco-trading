from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Advanced feature engineering for ML-based trading."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.feature_groups = self.config.get("feature_groups", [
            "price_returns",
            "volatility",
            "momentum",
            "volume",
            "pattern",
            "time",
        ])
        self.windows = self.config.get("windows", [5, 10, 20, 50])

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all features from OHLCV data."""
        if df.empty:
            return df

        result = df.copy()

        if "price_returns" in self.feature_groups:
            result = self._add_return_features(result)

        if "volatility" in self.feature_groups:
            result = self._add_volatility_features(result)

        if "momentum" in self.feature_groups:
            result = self._add_momentum_features(result)

        if "volume" in self.feature_groups:
            result = self._add_volume_features(result)

        if "pattern" in self.feature_groups:
            result = self._add_pattern_features(result)

        if "time" in self.feature_groups:
            result = self._add_time_features(result)

        result = self._clean_features(result)
        return result

    def _add_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return-based features."""
        for window in self.windows:
            if len(df) >= window:
                df[f"return_{window}"] = df["close"].pct_change(window)
                df[f"log_return_{window}"] = np.log(df["close"] / df["close"].shift(window))

        df["return_1"] = df["close"].pct_change()
        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features."""
        for window in self.windows:
            if len(df) >= window:
                df[f"volatility_{window}"] = df["return_1"].rolling(window).std()
                df[f"atr_ratio_{window}"] = (
                    (df["high"] - df["low"]).rolling(window).mean() / df["close"]
                )

        if len(df) >= 14:
            df["volatility_bb"] = (
                (df["close"] - df["close"].rolling(20).mean()) / (2 * df["close"].rolling(20).std())
            )

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        for window in self.windows:
            if len(df) >= window:
                df[f"rsi_{window}"] = self._calculate_rsi(df["close"], window)
                df[f"momentum_{window}"] = df["close"] - df["close"].shift(window)
                df[f"roc_{window}"] = (df["close"] - df["close"].shift(window)) / df["close"].shift(window) * 100

        if len(df) >= 26:
            df["macd"] = df["close"].ewm(span=12, adjust=False).mean() - df["close"].ewm(span=26, adjust=False).mean()
            df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
            df["macd_hist"] = df["macd"] - df["macd_signal"]

        return df

    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        if "volume" not in df.columns:
            return df

        for window in self.windows:
            if len(df) >= window:
                df[f"volume_ma_{window}"] = df["volume"].rolling(window).mean()
                df[f"volume_ratio_{window}"] = df["volume"] / df[f"volume_ma_{window}"]

        if len(df) >= 20:
            df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).cumsum()
            df["obv_ma"] = df["obv"].rolling(20).mean()

        return df

    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add chart pattern features."""
        if len(df) >= 20:
            df["high_max"] = df["high"].rolling(20).max()
            df["low_min"] = df["low"].rolling(20).min()
            df["range_position"] = (df["close"] - df["low_min"]) / (df["high_max"] - df["low_min"]).replace(0, np.nan)

        if len(df) >= 50:
            df["sma_20"] = df["close"].rolling(20).mean()
            df["sma_50"] = df["close"].rolling(50).mean()
            df["sma_cross"] = np.where(df["sma_20"] > df["sma_50"], 1, -1)

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], utc=True)
            df["hour"] = ts.dt.hour
            df["day_of_week"] = ts.dt.dayofweek
            df["day_of_month"] = ts.dt.day
            df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

        return df

    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare features."""
        df = df.replace([np.inf, -np.inf], np.nan)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(method="ffill").fillna(0)
        return df

    def get_feature_names(self) -> list[str]:
        """Get list of all feature names."""
        features = []
        for group in self.feature_groups:
            if group == "price_returns":
                features.extend([f"return_{w}" for w in self.windows])
                features.extend([f"log_return_{w}" for w in self.windows])
            elif group == "volatility":
                features.extend([f"volatility_{w}" for w in self.windows])
                features.extend([f"atr_ratio_{w}" for w in self.windows])
            elif group == "momentum":
                features.extend([f"rsi_{w}" for w in self.windows])
                features.extend([f"momentum_{w}" for w in self.windows])
            elif group == "volume":
                features.extend([f"volume_ma_{w}" for w in self.windows])
                features.extend([f"volume_ratio_{w}" for w in self.windows])
        return features
