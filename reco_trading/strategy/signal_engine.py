from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from reco_trading.strategy.order_flow import OrderFlowAnalyzer, OrderFlowDecision
from reco_trading.strategy.regime_filter import RegimeDecision, RegimeFilter

SignalValue = str


@dataclass(slots=True)
class SignalBundle:
    trend: SignalValue
    momentum: SignalValue
    volume: SignalValue
    volatility: SignalValue
    structure: SignalValue
    order_flow: SignalValue
    regime: str
    regime_trade_allowed: bool
    size_multiplier: float
    atr_ratio: float
    reversal_confirmed: bool
    dip_detected: bool
    liquidity_ok: bool
    support_zone: float
    resistance_zone: float


class SignalEngine:
    """Multi-factor signal generation."""

    def __init__(self) -> None:
        self.regime_filter = RegimeFilter()
        self.order_flow_analyzer = OrderFlowAnalyzer()

    def generate(self, df5m: pd.DataFrame, df15m: pd.DataFrame) -> SignalBundle:
        if len(df5m) < 3 or len(df15m) < 1:
            raise ValueError("insufficient_market_data_for_signal_generation")

        row = df5m.iloc[-1]
        prev = df5m.iloc[-2]
        prev2 = df5m.iloc[-3]
        confirm = df15m.iloc[-1]

        trend = "BUY" if row["ema20"] > row["ema50"] and confirm["ema20"] > confirm["ema50"] else "SELL"
        momentum = "BUY" if row["rsi"] > 55 else "SELL" if row["rsi"] < 45 else "NEUTRAL"
        volume = "BUY" if row["volume"] > row["vol_ma20"] * 1.1 else "NEUTRAL"
        higher_high = row["high"] > prev["high"] and row["low"] > prev["low"]
        lower_low = row["high"] < prev["high"] and row["low"] < prev["low"]
        structure = "BUY" if higher_high else "SELL" if lower_low else "NEUTRAL"

        regime_decision: RegimeDecision = self.regime_filter.evaluate(df5m)
        order_flow_decision: OrderFlowDecision = self.order_flow_analyzer.evaluate(df5m)

        volatility = "BUY" if regime_decision.allow_trade else "NEUTRAL"
        support_zone, resistance_zone = self._liquidity_zones(df5m)

        price = float(row["close"])
        buying_dip = self._is_dip(df5m)
        reversal_confirmed = self._reversal_confirmed(df5m)

        directional_signal = trend if trend in {"BUY", "SELL"} else structure
        liquidity_ok = True
        if directional_signal == "BUY" and price >= resistance_zone * 0.997:
            liquidity_ok = False
        if directional_signal == "SELL" and price <= support_zone * 1.003:
            liquidity_ok = False

        if directional_signal == "BUY":
            price_falling = float(row["close"]) < float(prev["close"]) < float(prev2["close"])
            reversal_confirmed = reversal_confirmed and (price_falling or buying_dip)
        elif directional_signal == "SELL":
            price_rising = float(row["close"]) > float(prev["close"]) > float(prev2["close"])
            reversal_confirmed = reversal_confirmed and price_rising
        else:
            reversal_confirmed = False

        return SignalBundle(
            trend=trend,
            momentum=momentum,
            volume=volume,
            volatility=volatility,
            structure=structure,
            order_flow=order_flow_decision.signal,
            regime=regime_decision.regime.value,
            regime_trade_allowed=regime_decision.allow_trade,
            size_multiplier=regime_decision.size_multiplier,
            atr_ratio=regime_decision.atr_ratio,
            reversal_confirmed=reversal_confirmed,
            dip_detected=buying_dip,
            liquidity_ok=liquidity_ok,
            support_zone=support_zone,
            resistance_zone=resistance_zone,
        )

    def _reversal_confirmed(self, frame: pd.DataFrame) -> bool:
        recent = frame.tail(8)
        if len(recent) < 4:
            return False

        rsi = recent["rsi"]
        close = recent["close"]
        ema_slope_now = float(recent["ema20"].iloc[-1] - recent["ema20"].iloc[-2])
        ema_slope_prev = float(recent["ema20"].iloc[-2] - recent["ema20"].iloc[-3])
        momentum_crossover = float(rsi.iloc[-1]) > float(rsi.iloc[-2]) and float(rsi.iloc[-1]) > 45
        bullish_structure = float(recent["close"].iloc[-1]) > float(recent["open"].iloc[-1]) and float(recent["low"].iloc[-1]) >= float(recent["low"].iloc[-2])
        bearish_structure = float(recent["close"].iloc[-1]) < float(recent["open"].iloc[-1]) and float(recent["high"].iloc[-1]) <= float(recent["high"].iloc[-2])
        rsi_divergence = (float(close.iloc[-1]) < float(close.iloc[-2]) and float(rsi.iloc[-1]) > float(rsi.iloc[-2])) or (
            float(close.iloc[-1]) > float(close.iloc[-2]) and float(rsi.iloc[-1]) < float(rsi.iloc[-2])
        )
        slope_change = (ema_slope_prev <= 0 < ema_slope_now) or (ema_slope_prev >= 0 > ema_slope_now)

        return bool((momentum_crossover and bullish_structure) or bearish_structure or rsi_divergence or slope_change)

    def _is_dip(self, frame: pd.DataFrame) -> bool:
        recent = frame.tail(20)
        if recent.empty:
            return False
        row = recent.iloc[-1]
        ma20 = float(row["ema20"])
        close = float(row["close"])
        if ma20 <= 0:
            return False
        below_ma = close <= ma20 * 0.985
        volume_spike = float(row["volume"]) >= float(row["vol_ma20"]) * 1.2
        momentum_slowdown = abs(float(recent["rsi"].iloc[-1] - recent["rsi"].iloc[-2])) < 4.0
        return below_ma and volume_spike and momentum_slowdown

    def _liquidity_zones(self, frame: pd.DataFrame) -> tuple[float, float]:
        recent = frame.tail(30)
        support = float(recent["low"].rolling(window=5).min().iloc[-1])
        resistance = float(recent["high"].rolling(window=5).max().iloc[-1])
        return support, resistance

    def is_sideways(self, df: pd.DataFrame) -> bool:
        if len(df) < 60:
            return True
        recent = df.tail(30)
        atr_ratio = recent["atr"].iloc[-1] / recent["close"].iloc[-1]
        ema_distance = abs(recent["ema20"].iloc[-1] - recent["ema50"].iloc[-1]) / recent["close"].iloc[-1]
        crossings = ((recent["ema20"] > recent["ema50"]).astype(int).diff().abs() == 1).sum()
        return atr_ratio < 0.003 or ema_distance < 0.001 or crossings >= 6
