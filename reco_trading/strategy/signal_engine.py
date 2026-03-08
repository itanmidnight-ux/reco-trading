from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


SignalValue = str


@dataclass(slots=True)
class SignalBundle:
    trend: SignalValue
    momentum: SignalValue
    volume: SignalValue
    volatility: SignalValue
    structure: SignalValue


class SignalEngine:
    """Multi-factor signal generation."""

    def generate(self, df5m: pd.DataFrame, df15m: pd.DataFrame) -> SignalBundle:
        row = df5m.iloc[-1]
        prev = df5m.iloc[-2]
        confirm = df15m.iloc[-1]

        trend = "BUY" if row["ema20"] > row["ema50"] and confirm["ema20"] > confirm["ema50"] else "SELL"
        momentum = "BUY" if row["rsi"] > 55 else "SELL" if row["rsi"] < 45 else "NEUTRAL"
        volume = "BUY" if row["volume"] > row["vol_ma20"] * 1.1 else "NEUTRAL"
        volatility = "BUY" if row["atr"] > prev["atr"] * 0.98 else "NEUTRAL"
        higher_high = row["high"] > prev["high"] and row["low"] > prev["low"]
        lower_low = row["high"] < prev["high"] and row["low"] < prev["low"]
        structure = "BUY" if higher_high else "SELL" if lower_low else "NEUTRAL"

        return SignalBundle(
            trend=trend,
            momentum=momentum,
            volume=volume,
            volatility=volatility,
            structure=structure,
        )

    def is_sideways(self, df: pd.DataFrame) -> bool:
        if len(df) < 60:
            return True
        recent = df.tail(30)
        atr_ratio = recent["atr"].iloc[-1] / recent["close"].iloc[-1]
        ema_distance = abs(recent["ema20"].iloc[-1] - recent["ema50"].iloc[-1]) / recent["close"].iloc[-1]
        crossings = ((recent["ema20"] > recent["ema50"]).astype(int).diff().abs() == 1).sum()
        return atr_ratio < 0.003 or ema_distance < 0.001 or crossings >= 6
