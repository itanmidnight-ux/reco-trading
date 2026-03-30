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

        ema20 = _safe_getitem(row, "ema20")
        ema50 = _safe_getitem(row, "ema50")
        confirm_ema20 = _safe_getitem(confirm, "ema20")
        confirm_ema50 = _safe_getitem(confirm, "ema50")
        
        primary_bull = ema20 > ema50
        confirm_bull = confirm_ema20 > confirm_ema50
        primary_bear = ema20 < ema50
        confirm_bear = confirm_ema20 < confirm_ema50
        if primary_bull and confirm_bull:
            trend = "BUY"
        elif primary_bear and confirm_bear:
            trend = "SELL"
        else:
            trend = "NEUTRAL"
        
        rsi = _safe_getitem(row, "rsi", 50.0)
        if rsi > 52:
            momentum = "BUY"
        elif rsi < 48:
            momentum = "SELL"
        else:
            momentum = "NEUTRAL"

        volume_val = _safe_getitem(row, "volume", 0.0)
        vol_ma20 = _safe_getitem(row, "vol_ma20", 1.0)
        vol_ratio = volume_val / max(vol_ma20, 1e-9)
        if vol_ratio > 1.00:
            volume = "BUY"
        elif vol_ratio < 0.85:
            volume = "SELL"
        else:
            volume = "NEUTRAL"

        high_val = _safe_getitem(row, "high", 0.0)
        low_val = _safe_getitem(row, "low", 0.0)
        prev_high = _safe_getitem(prev, "high", 0.0)
        prev_low = _safe_getitem(prev, "low", 0.0)
        
        higher_high = high_val > prev_high and low_val > prev_low
        lower_low = high_val < prev_high and low_val < prev_low
        structure = "BUY" if higher_high else "SELL" if lower_low else "NEUTRAL"

        if len(df5m) >= 4:
            prev2 = df5m.iloc[-3]
            close_val = _safe_getitem(row, "close", 0.0)
            prev2_close = _safe_getitem(prev2, "close", 0.0)
            micro_up = close_val > prev2_close
            micro_down = close_val < prev2_close
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
        
        adx = _safe_float(row.get("adx"), 20.0)
        di_plus = _safe_float(row.get("di_plus"), 0.0)
        di_minus = _safe_float(row.get("di_minus"), 0.0)
        
        trend_strength = "STRONG" if adx >= 20 else "MODERATE" if adx >= 15 else "WEAK"
        if trend == "BUY" and di_plus > di_minus and adx >= 12:
            trend = "BUY"
        elif trend == "SELL" and di_minus > di_plus and adx >= 12:
            trend = "SELL"
        elif adx < 10:
            trend = "NEUTRAL"

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
        if hasattr(value, "iloc"):
            value = value.iloc[-1] if len(value) > 0 else default
        elif hasattr(value, "item"):
            value = value.item()
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _safe_getitem(row: pd.Series, key: str, default: float = 0.0) -> float:
    """Safely get a scalar value from a pandas Series."""
    try:
        val = row[key]
        if hasattr(val, "iloc"):
            val = val.iloc[-1] if len(val) > 0 else default
        elif hasattr(val, "item"):
            val = val.item()
        return float(val)
    except (TypeError, ValueError, KeyError):
        return default
