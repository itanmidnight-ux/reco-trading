from __future__ import annotations

import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator, CCIIndicator, EMAIndicator, MACD as MACDIndicator
from ta.volatility import AverageTrueRange


def apply_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Enrich OHLCV DataFrame with required indicators for signal and safety filters."""
    df = frame.copy()
    df["ema20"] = EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema50"] = EMAIndicator(df["close"], window=50).ema_indicator()
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df["adx"] = ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()
    df["vol_ma20"] = df["volume"].rolling(window=20).mean()

    df["ema9"] = EMAIndicator(df["close"], window=9).ema_indicator()

    macd_ind = MACDIndicator(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_diff"] = macd_ind.macd_diff()

    df["cci"] = CCIIndicator(df["high"], df["low"], df["close"], window=20).cci()

    stoch = StochasticOscillator(df["high"], df["low"], df["close"], window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    return df.dropna().reset_index(drop=True)
