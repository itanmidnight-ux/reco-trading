from __future__ import annotations

import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator, CCIIndicator, EMAIndicator, MACD as MACDIndicator
from ta.volatility import AverageTrueRange


def apply_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Enrich OHLCV DataFrame with required indicators for signal and safety filters."""
    df = frame.copy()
    candle_range = (df["high"] - df["low"]).clip(lower=1e-9)
    body = df["close"] - df["open"]
    body_abs = body.abs()
    upper_wick = (df["high"] - df[["open", "close"]].max(axis=1)).clip(lower=0.0)
    lower_wick = (df[["open", "close"]].min(axis=1) - df["low"]).clip(lower=0.0)

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

    df["body_size_abs"] = body_abs
    df["body_ratio"] = body_abs / candle_range
    df["upper_wick"] = upper_wick
    df["lower_wick"] = lower_wick
    df["wick_ratio"] = (upper_wick + lower_wick) / candle_range
    df["is_pin_bar"] = (
        (df["wick_ratio"] >= 0.65)
        & (df["body_ratio"] <= 0.35)
        & ((upper_wick >= (body_abs * 1.5)) | (lower_wick >= (body_abs * 1.5)))
    )
    df["is_hammer"] = (lower_wick >= (body_abs * 2.0)) & (upper_wick <= (body_abs + 1e-9)) & (body > 0)
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    prev_body_bear = prev_close < prev_open
    prev_body_bull = prev_close > prev_open
    df["engulfing_bull"] = prev_body_bear & (df["close"] > df["open"]) & (df["open"] <= prev_close) & (df["close"] >= prev_open)
    df["engulfing_bear"] = prev_body_bull & (df["close"] < df["open"]) & (df["open"] >= prev_close) & (df["close"] <= prev_open)
    df["is_doji"] = df["body_ratio"] <= 0.10
    df["close_location"] = ((df["close"] - df["low"]) / candle_range).clip(lower=0.0, upper=1.0)
    return df.dropna().reset_index(drop=True)
