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
        if len(df5m) < 2 or len(df15m) < 2:
            return SignalBundle(
                trend="NEUTRAL",
                momentum="NEUTRAL",
                volume="NEUTRAL",
                volatility="NEUTRAL",
                structure="NEUTRAL",
                order_flow="NEUTRAL",
                regime="INSUFFICIENT_DATA",
                regime_trade_allowed=False,
                size_multiplier=0.0,
                atr_ratio=0.0,
            )

        row = df5m.iloc[-1]
        prev = df5m.iloc[-2]
        confirm = df15m.iloc[-1]

        primary_bull = row["ema20"] > row["ema50"]
        confirm_bull = confirm["ema20"] > confirm["ema50"]
        primary_bear = row["ema20"] < row["ema50"]
        confirm_bear = confirm["ema20"] < confirm["ema50"]
        if primary_bull and confirm_bull:
            trend = "BUY"
        elif primary_bear and confirm_bear:
            trend = "SELL"
        else:
            trend = "NEUTRAL"
        if row["rsi"] > 51:
            momentum = "BUY"
        elif row["rsi"] < 49:
            momentum = "SELL"
        else:
            momentum = "NEUTRAL"

        vol_ratio = row["volume"] / max(row["vol_ma20"], 1e-9)
        if vol_ratio > 1.02:
            volume = "BUY"
        elif vol_ratio < 0.70:
            volume = "SELL"
        else:
            volume = "NEUTRAL"

        higher_high = row["high"] > prev["high"] and row["low"] > prev["low"]
        lower_low = row["high"] < prev["high"] and row["low"] < prev["low"]
        structure = "BUY" if higher_high else "SELL" if lower_low else "NEUTRAL"

        if len(df5m) >= 4:
            prev2 = df5m.iloc[-3]
            micro_up = row["close"] > prev2["close"]
            micro_down = row["close"] < prev2["close"]
            if higher_high or (not lower_low and micro_up):
                structure = "BUY"
            elif lower_low or (not higher_high and micro_down):
                structure = "SELL"
            else:
                structure = "NEUTRAL"

        macd_diff_now = _safe_float(row.get("macd_diff"), 0.0)
        macd_diff_prev = _safe_float(prev.get("macd_diff"), 0.0)
        macd_cross_up = macd_diff_now > 0 and macd_diff_prev <= 0
        macd_cross_down = macd_diff_now < 0 and macd_diff_prev >= 0

        stoch_k = _safe_float(row.get("stoch_k"), 50.0)
        stoch_oversold = stoch_k < 25
        stoch_overbought = stoch_k > 75

        if momentum == "BUY" and macd_cross_up:
            momentum = "BUY"
        elif momentum == "NEUTRAL" and macd_cross_up and not stoch_overbought:
            momentum = "BUY"
        elif momentum == "NEUTRAL" and macd_cross_down and not stoch_oversold:
            momentum = "SELL"

        regime_decision: RegimeDecision = self.regime_filter.evaluate(df5m)
        order_flow_decision: OrderFlowDecision = self.order_flow_analyzer.evaluate(df5m)

        volatility = "BUY" if regime_decision.allow_trade else "NEUTRAL"

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

    def is_sideways(self, df: pd.DataFrame) -> bool:
        if len(df) < 60:
            return True
        recent = df.tail(30)
        atr_ratio = recent["atr"].iloc[-1] / recent["close"].iloc[-1]
        ema_distance = abs(recent["ema20"].iloc[-1] - recent["ema50"].iloc[-1]) / recent["close"].iloc[-1]
        crossings = ((recent["ema20"] > recent["ema50"]).astype(int).diff().abs() == 1).sum()
        return atr_ratio < 0.003 or ema_distance < 0.001 or crossings >= 6


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
