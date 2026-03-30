from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Any

from reco_trading.risk.position_manager import Position


def _sigmoid(value: float) -> float:
    capped = max(min(float(value), 30.0), -30.0)
    return 1.0 / (1.0 + exp(-capped))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        import pandas as pd
        if isinstance(value, pd.Series):
            value = value.iloc[-1] if len(value) > 0 else default
        elif isinstance(value, pd.DataFrame):
            value = value.iloc[-1, -1] if value.size > 0 else default
        return float(value)
    except (TypeError, ValueError, IndexError):
        return default


@dataclass(slots=True)
class ExitIntelligenceDecision:
    exit_now: bool
    score: float
    threshold: float
    reason: str
    reason_codes: tuple[str, ...]
    details: dict[str, float]


class ExitIntelligence:
    """Adaptive, score-based profit protection and graceful trade exit helper."""

    def evaluate(
        self,
        *,
        position: Position,
        market_data: dict[str, Any],
        current_price: float,
        atr: float,
    ) -> ExitIntelligenceDecision:
        safe_price = max(float(current_price), 1e-9)
        used_atr = max(float(atr), safe_price * 0.0015, 1e-9)
        risk_distance = max(float(position.initial_risk_distance), used_atr * 0.6, 1e-9)

        if position.side == "BUY":
            open_profit = safe_price - position.entry_price
            peak_profit = max((position.peak_price or position.entry_price) - position.entry_price, 0.0)
        else:
            open_profit = position.entry_price - safe_price
            peak_profit = max(position.entry_price - (position.peak_price or position.entry_price), 0.0)

        open_profit_r = max(open_profit / risk_distance, -3.0)
        peak_profit_r = max(peak_profit / risk_distance, 0.0)
        giveback_r = max(peak_profit_r - max(open_profit_r, 0.0), 0.0)

        atr_ratio = used_atr / safe_price
        threshold = self._dynamic_threshold(atr_ratio)

        frame5 = market_data.get("frame5")
        ema20_slope = 0.0
        momentum_fade = 0.0
        structure_break = 0.0
        if frame5 is not None and hasattr(frame5, "iloc") and len(frame5) >= 4:
            row = frame5.iloc[-1]
            prev = frame5.iloc[-2]
            prev2 = frame5.iloc[-3]
            prev3 = frame5.iloc[-4]
            ema20_now = _safe_float(row.get("ema20"), safe_price)
            ema20_prev = _safe_float(prev3.get("ema20"), ema20_now)
            ema20_slope = (ema20_now - ema20_prev) / max(abs(ema20_prev), 1e-9)
            if position.side == "BUY":
                rsi_now = _safe_float(row.get("rsi"), 50.0)
                rsi_prev = _safe_float(prev.get("rsi"), rsi_now)
                macd_now = _safe_float(row.get("macd_diff"), 0.0)
                macd_prev = _safe_float(prev.get("macd_diff"), macd_now)
                momentum_fade = max((rsi_prev - rsi_now) / 40.0, 0.0) + max((macd_prev - macd_now) / max(abs(macd_prev), 1.0), 0.0) * 0.5
                bearish_shift = _safe_float(row.get("close"), safe_price) < _safe_float(prev.get("close"), safe_price) < _safe_float(prev2.get("close"), safe_price)
                structure_break = 1.0 if bearish_shift and ema20_slope <= 0 else 0.0
            else:
                rsi_now = _safe_float(row.get("rsi"), 50.0)
                rsi_prev = _safe_float(prev.get("rsi"), rsi_now)
                macd_now = _safe_float(row.get("macd_diff"), 0.0)
                macd_prev = _safe_float(prev.get("macd_diff"), macd_now)
                momentum_fade = max((rsi_now - rsi_prev) / 40.0, 0.0) + max((macd_now - macd_prev) / max(abs(macd_prev), 1.0), 0.0) * 0.5
                bullish_shift = _safe_float(row.get("close"), safe_price) > _safe_float(prev.get("close"), safe_price) > _safe_float(prev2.get("close"), safe_price)
                structure_break = 1.0 if bullish_shift and ema20_slope >= 0 else 0.0

        expected_move = used_atr / safe_price
        spread_ratio = max(_safe_float(market_data.get("spread"), 0.0), 0.0) / safe_price
        cost_pressure = spread_ratio / max(expected_move, 1e-9)

        bars_held = int(getattr(position, "bars_held", 0))
        time_pressure = max((bars_held - 18) / 18.0, 0.0)

        giveback_score = _sigmoid((giveback_r - 0.45) / 0.22)
        momentum_score = _sigmoid(momentum_fade * 2.2)
        structure_score = _sigmoid((structure_break * 2.0) + (max(-ema20_slope if position.side == "BUY" else ema20_slope, 0.0) * 80.0))
        time_score = _sigmoid(time_pressure)
        cost_score = _sigmoid((cost_pressure - 0.45) / 0.30)
        score = (
            0.32 * giveback_score
            + 0.22 * momentum_score
            + 0.24 * structure_score
            + 0.12 * time_score
            + 0.10 * cost_score
        )

        reason_codes: list[str] = []
        if giveback_r >= 0.35:
            reason_codes.append("GIVEBACK")
        if momentum_fade >= 0.40:
            reason_codes.append("MOMENTUM_FADE")
        if structure_break >= 0.8:
            reason_codes.append("STRUCTURE_WEAKNESS")
        if bars_held >= 24 and open_profit_r < 0.30:
            reason_codes.append("TIME_EFFICIENCY")
        if cost_pressure >= 0.90:
            reason_codes.append("COST_PRESSURE")

        exit_profitable = open_profit_r >= 0.35
        exit_now = bool(exit_profitable and score >= threshold and len(reason_codes) >= 1)
        reason = "EXIT_INTELLIGENCE_HOLD"
        if exit_now:
            reason = f"EXIT_INTELLIGENCE_{reason_codes[0]}"

        return ExitIntelligenceDecision(
            exit_now=exit_now,
            score=float(round(score, 4)),
            threshold=float(round(threshold, 4)),
            reason=reason,
            reason_codes=tuple(reason_codes[:3]),
            details={
                "open_profit_r": round(open_profit_r, 4),
                "peak_profit_r": round(peak_profit_r, 4),
                "giveback_r": round(giveback_r, 4),
                "atr_ratio": round(atr_ratio, 6),
                "ema20_slope": round(ema20_slope, 6),
                "momentum_fade": round(momentum_fade, 4),
                "structure_break": round(structure_break, 4),
                "time_pressure": round(time_pressure, 4),
                "cost_pressure": round(cost_pressure, 4),
            },
        )

    @staticmethod
    def _dynamic_threshold(atr_ratio: float) -> float:
        if atr_ratio < 0.004:
            return 0.58
        if atr_ratio < 0.012:
            return 0.65
        return 0.72
