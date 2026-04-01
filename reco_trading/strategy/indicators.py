from __future__ import annotations

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, UltimateOscillator, WilliamsRIndicator
from ta.trend import ADXIndicator, CCIIndicator, EMAIndicator, MACD as MACDIndicator, IchimokuIndicator, KSTIndicator
from ta.volatility import AverageTrueRange, BollingerBands, KeltnerChannel, DonchianChannel
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator


def _accumulation_distribution(high, low, close, volume):
    """Calculate Accumulation/Distribution Line."""
    clv = ((close - low) - (high - close)) / (high - low).replace(0, 1e-9)
    clv = clv.fillna(0)
    return (clv * volume).cumsum()


def _chaikin_money_flow(high, low, close, volume, window=20):
    """Calculate Chaikin Money Flow."""
    ad = _accumulation_distribution(high, low, close, volume)
    return ad.rolling(window).mean() / volume.rolling(window).sum()


def apply_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Enrich OHLCV DataFrame with required indicators for signal and safety filters."""
    df = frame.copy()
    candle_range = (df["high"] - df["low"]).clip(lower=1e-9)
    body = df["close"] - df["open"]
    body_abs = body.abs()
    upper_wick = (df["high"] - df[["open", "close"]].max(axis=1)).clip(lower=0.0)
    lower_wick = (df[["open", "close"]].min(axis=1) - df["low"]).clip(lower=0.0)

    # Trend indicators
    df["ema9"] = EMAIndicator(df["close"], window=9).ema_indicator()
    df["ema20"] = EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema50"] = EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema100"] = EMAIndicator(df["close"], window=100).ema_indicator()
    df["ema200"] = EMAIndicator(df["close"], window=200).ema_indicator()
    
    # MACD
    macd_ind = MACDIndicator(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_diff"] = macd_ind.macd_diff()
    
    # Momentum
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
    df["rsi_28"] = RSIIndicator(df["close"], window=28).rsi()
    df["roc"] = ROCIndicator(df["close"], window=12).roc()
    df["stoch_k"] = StochasticOscillator(df["high"], df["low"], df["close"], window=14, smooth_window=3).stoch()
    df["stoch_d"] = StochasticOscillator(df["high"], df["low"], df["close"], window=14, smooth_window=3).stoch_signal()
    df["ultimate_osc"] = UltimateOscillator(df["high"], df["low"], df["close"]).ultimate_oscillator()
    df["williams_r"] = WilliamsRIndicator(df["high"], df["low"], df["close"]).williams_r()
    
    # Trend strength
    df["adx"] = ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()
    df["adx_neg"] = ADXIndicator(df["high"], df["low"], df["close"], window=14).adx_neg()
    df["adx_pos"] = ADXIndicator(df["high"], df["low"], df["close"], window=14).adx_pos()
    df["cci"] = CCIIndicator(df["high"], df["low"], df["close"], window=20).cci()
    
    # Volatility
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df["atr_pct"] = (df["atr"] / df["close"]) * 100
    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    
    # Volume
    df["vol_ma20"] = df["volume"].rolling(window=20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma20"]
    df["vwap"] = VolumeWeightedAveragePrice(df["high"], df["low"], df["close"], df["volume"]).volume_weighted_average_price()
    df["obv"] = OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
    df["cmf"] = _chaikin_money_flow(df["high"], df["low"], df["close"], df["volume"])
    df["adi"] = _accumulation_distribution(df["high"], df["low"], df["close"], df["volume"])
    
    # Support/Resistance
    df["donchian_upper"] = DonchianChannel(df["high"], df["low"], df["close"], window=20).donchian_channel_hband()
    df["donchian_lower"] = DonchianChannel(df["high"], df["low"], df["close"], window=20).donchian_channel_lband()
    df["donchian_mid"] = DonchianChannel(df["high"], df["low"], df["close"], window=20).donchian_channel_mband()
    
    # Ichimoku
    try:
        ichimoku = IchimokuIndicator(df["high"], df["low"], df["close"])
        df["ichimoku_a"] = ichimoku.ichimoku_a()
        df["ichimoku_b"] = ichimoku.ichimoku_b()
        df["ichimoku_base"] = ichimoku.ichimoku_base_line()
        df["ichimoku_conv"] = ichimoku.ichimoku_conversion_line()
    except:
        df["ichimoku_a"] = df["close"]
        df["ichimoku_b"] = df["close"]
        df["ichimoku_base"] = df["close"]
        df["ichimoku_conv"] = df["close"]
    
    # KST (Know Sure Thing)
    kst = KSTIndicator(df["close"])
    df["kst"] = kst.kst()
    df["kst_signal"] = kst.kst_sig()
    
    # Pattern recognition
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
    
    # Moving averages cross
    df["ema_cross_up"] = (df["ema9"] > df["ema20"]) & (df["ema9"].shift(1) <= df["ema20"].shift(1))
    df["ema_cross_down"] = (df["ema9"] < df["ema20"]) & (df["ema9"].shift(1) >= df["ema20"].shift(1))
    
    # Price position relative to BB
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill().fillna(0.0)
    return df.reset_index(drop=True)
