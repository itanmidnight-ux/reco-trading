from __future__ import annotations

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange


def apply_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Enrich a OHLCV DataFrame with required indicators."""
    df = frame.copy()
    df["ema20"] = EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema50"] = EMAIndicator(df["close"], window=50).ema_indicator()
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df["vol_ma20"] = df["volume"].rolling(window=20).mean()
    return df.dropna().reset_index(drop=True)
