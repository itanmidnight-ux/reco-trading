from __future__ import annotations

from trading_system.app.services.feature_engineering.pipeline import FeatureVector


def ml_features(f: FeatureVector, deterministic_score: float) -> list[float]:
    return [
        f.rsi / 100,
        f.macd_hist,
        f.ema9 - f.ema21,
        f.atr,
        f.orderbook_imbalance,
        f.volatility,
        f.breakout_score,
        deterministic_score,
    ]
