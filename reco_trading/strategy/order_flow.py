from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class OrderFlowDecision:
    signal: str
    buy_pressure: float
    sell_pressure: float


class OrderFlowAnalyzer:
    """Approximates order flow pressure from candle direction and volume."""

    def __init__(self, lookback: int = 20) -> None:
        self.lookback = lookback

    def evaluate(self, frame: pd.DataFrame) -> OrderFlowDecision:
        recent = frame.tail(self.lookback)
        if recent.empty:
            return OrderFlowDecision(signal="NEUTRAL", buy_pressure=0.5, sell_pressure=0.5)

        buy_volume = float(recent.loc[recent["close"] > recent["open"], "volume"].sum())
        sell_volume = float(recent.loc[recent["close"] < recent["open"], "volume"].sum())
        total = buy_volume + sell_volume
        if total <= 0:
            return OrderFlowDecision(signal="NEUTRAL", buy_pressure=0.5, sell_pressure=0.5)

        buy_pressure = buy_volume / total
        sell_pressure = sell_volume / total

        if buy_pressure > 0.60:
            signal = "BUY"
        elif buy_pressure < 0.40:
            signal = "SELL"
        else:
            signal = "NEUTRAL"
        return OrderFlowDecision(signal=signal, buy_pressure=buy_pressure, sell_pressure=sell_pressure)
