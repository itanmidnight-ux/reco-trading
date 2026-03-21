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

    # ============ NUEVO: CANDLE STRUCTURE ANALYSIS ============
    # Análisis de estructura de velas para detección de patrones
    atr_safe = df["atr"].clip(lower=1e-9)

    # Body size: tamaño del cuerpo normalizado por ATR
    df["body_size_abs"] = (df["close"] - df["open"]).abs()
    df["body_ratio"] = df["body_size_abs"] / atr_safe

    # Wick sizes: tamaños de mechas (wicks)
    # Upper wick: desde cierre/apertura más alta hasta máximo
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    # Lower wick: desde cierre/apertura más baja hasta mínimo
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    # Total wick ratio
    df["wick_ratio"] = (df["upper_wick"] + df["lower_wick"]) / atr_safe

    # Pin bar: cuerpo pequeño + wick grande (reversal pattern)
    df["is_pin_bar"] = (df["wick_ratio"] > 2.0) & (df["body_ratio"] < 0.5)

    # Ratios individuales para hammer detection
    df["upper_wick_ratio"] = df["upper_wick"] / atr_safe
    df["lower_wick_ratio"] = df["lower_wick"] / atr_safe

    # Hammer: cuerpo pequeño arriba + wick grande abajo (reversal alcista)
    df["is_hammer"] = (
        (df["body_ratio"] < 0.5)
        & (df["lower_wick_ratio"] > 2.0)
        & (df["lower_wick_ratio"] > df["upper_wick_ratio"] * 2)
    )

    # Engulfing patterns: vela actual envuelve la anterior
    df["prev_open"] = df["open"].shift(1)
    df["prev_close"] = df["close"].shift(1)

    # Engulfing alcista: abre dentro del anterior, cierra por fuera (arriba)
    df["engulfing_bull"] = (
        (df["open"] < df["prev_close"])
        & (df["close"] > df["prev_open"])
        & (df["close"] > df["prev_open"])
    )

    # Engulfing bajista: abre dentro del anterior, cierra por fuera (abajo)
    df["engulfing_bear"] = (
        (df["open"] > df["prev_close"])
        & (df["close"] < df["prev_open"])
        & (df["close"] < df["prev_open"])
    )

    # Doji: indecisión del mercado (cuerpo muy pequeño, wicks similares)
    df["is_doji"] = (
        (df["body_ratio"] < 0.1)
        & ((df["upper_wick_ratio"] - df["lower_wick_ratio"]).abs() < 0.5)
    )

    # Close location: posición del cierre dentro del rango día
    df["close_location"] = (df["close"] - df["low"]) / ((df["high"] - df["low"]).clip(lower=1e-9))

    # ============ FIN CANDLE STRUCTURE ANALYSIS ============
    return df.dropna().reset_index(drop=True)
