from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from reco_trading.strategy.order_flow import OrderFlowAnalyzer, OrderFlowDecision
from reco_trading.strategy.regime_filter import RegimeDecision, RegimeFilter, VolatilityRegime

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
    reversal_confirmed: bool = False
    dip_detected: bool = False
    liquidity_ok: bool = True
    support_zone: float = 0.0
    resistance_zone: float = 0.0


class SignalEngine:
    """Multi-factor signal generation."""

    def __init__(self) -> None:
        self.regime_filter = RegimeFilter()
        self.order_flow_analyzer = OrderFlowAnalyzer()

    def generate(self, df5m: pd.DataFrame, df15m: pd.DataFrame) -> SignalBundle:
        if len(df5m) < 2 or len(df15m) < 1:
            return self._neutral_bundle()

        row = df5m.iloc[-1]
        prev = df5m.iloc[-2]
        confirm = df15m.iloc[-1]

        if row["ema20"] > row["ema50"] and confirm["ema20"] > confirm["ema50"]:
            trend = "BUY"
        elif row["ema20"] < row["ema50"] and confirm["ema20"] < confirm["ema50"]:
            trend = "SELL"
        else:
            trend = "NEUTRAL"
        momentum = "BUY" if row["rsi"] > 55 else "SELL" if row["rsi"] < 45 else "NEUTRAL"
        if row["vol_ma20"] > 0 and row["volume"] > row["vol_ma20"] * 1.1:
            if row["close"] > row["open"]:
                volume = "BUY"
            elif row["close"] < row["open"]:
                volume = "SELL"
            else:
                volume = "NEUTRAL"
        else:
            volume = "NEUTRAL"
        higher_high = row["high"] > prev["high"] and row["low"] > prev["low"]
        lower_low = row["high"] < prev["high"] and row["low"] < prev["low"]
        structure = "BUY" if higher_high else "SELL" if lower_low else "NEUTRAL"

        regime_decision: RegimeDecision = self.regime_filter.evaluate(df5m)
        order_flow_decision: OrderFlowDecision = self.order_flow_analyzer.evaluate(df5m)

        volatility = "NEUTRAL"

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
        )

    @staticmethod
    def _neutral_bundle() -> SignalBundle:
        return SignalBundle(
            trend="NEUTRAL",
            momentum="NEUTRAL",
            volume="NEUTRAL",
            volatility="NEUTRAL",
            structure="NEUTRAL",
            order_flow="NEUTRAL",
            regime=VolatilityRegime.LOW_VOLATILITY.value,
            regime_trade_allowed=False,
            size_multiplier=0.0,
            atr_ratio=0.0,
        )

    def is_sideways(self, df: pd.DataFrame) -> bool:
        if len(df) < 60:
            return True
        recent = df.tail(30)
        atr_ratio = recent["atr"].iloc[-1] / recent["close"].iloc[-1]
        ema_distance = abs(recent["ema20"].iloc[-1] - recent["ema50"].iloc[-1]) / recent["close"].iloc[-1]
        crossings = ((recent["ema20"] > recent["ema50"]).astype(int).diff().abs() == 1).sum()
        return atr_ratio < 0.003 or ema_distance < 0.001 or crossings >= 6
