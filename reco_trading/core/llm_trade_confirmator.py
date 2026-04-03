from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class TradeConfirmation:
    symbol: str
    side: str
    entry_price: float
    quantity: float
    signal: str
    confidence: float
    confirmed: bool = False
    reason: str = ""
    analysis_time_ms: float = 0.0
    timestamp: str = ""


class LLMTradeConfirmator:
    """Ultra-fast trade confirmation using rule-based analysis (simulates LLM in <5ms)."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self._confirmation_count = 0
        self._rejection_count = 0
        self._avg_time_ms = 0.0

    def confirm_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        signal: str,
        confidence: float,
        trend: str = "NEUTRAL",
        adx: float = 0.0,
        volatility_regime: str = "NORMAL",
        order_flow: str = "NEUTRAL",
        spread: float = 0.0,
        atr: float = 0.0,
        volume: float = 0.0,
        risk_per_trade: float = 0.01,
        daily_pnl: float = 0.0,
        trades_today: int = 0,
        max_trades_per_day: int = 20,
    ) -> TradeConfirmation:
        start = time.perf_counter()

        score = 0.0
        reasons: list[str] = []

        if confidence >= 0.75:
            score += 25
        elif confidence >= 0.60:
            score += 15
        elif confidence >= 0.50:
            score += 5
        else:
            reasons.append(f"Low confidence: {confidence:.2f}")

        if trend in ("BULLISH", "BEARISH"):
            score += 15
            if (side == "BUY" and trend == "BULLISH") or (side == "SELL" and trend == "BEARISH"):
                score += 10
            else:
                score -= 10
                reasons.append("Trend/side mismatch")
        else:
            reasons.append("Neutral trend")

        if adx >= 25:
            score += 10
        elif adx >= 20:
            score += 5
        else:
            reasons.append("Weak trend strength (ADX low)")

        if volatility_regime in ("NORMAL", "TRENDING"):
            score += 10
        elif volatility_regime == "HIGH_VOLATILITY":
            score += 5
            reasons.append("High volatility - reduced size recommended")
        else:
            score -= 10
            reasons.append(f"Unfavorable regime: {volatility_regime}")

        if order_flow in ("BULLISH", "BEARISH"):
            score += 10
            if (side == "BUY" and order_flow == "BULLISH") or (side == "SELL" and order_flow == "BEARISH"):
                score += 5
        else:
            reasons.append("Neutral order flow")

        if spread < 0.001:
            score += 5
        elif spread < 0.005:
            score += 3
        else:
            score -= 5
            reasons.append(f"Wide spread: {spread:.4f}")

        if atr > 0:
            atr_pct = (atr / entry_price) * 100 if entry_price > 0 else 0
            if 0.5 <= atr_pct <= 5.0:
                score += 5
            elif atr_pct > 5.0:
                reasons.append(f"Extreme ATR: {atr_pct:.1f}%")

        if volume > 1000:
            score += 5

        if trades_today >= max_trades_per_day:
            score -= 30
            reasons.append(f"Daily trade limit reached: {trades_today}/{max_trades_per_day}")

        risk_amount = entry_price * quantity * risk_per_trade
        if risk_amount > 100:
            score -= 5
            reasons.append(f"High risk amount: ${risk_amount:.2f}")

        if daily_pnl < -50:
            score -= 15
            reasons.append(f"Significant daily loss: ${daily_pnl:.2f}")

        confirmed = score >= 50
        analysis_time = (time.perf_counter() - start) * 1000

        self._confirmation_count += 1
        if not confirmed:
            self._rejection_count += 1

        result = TradeConfirmation(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            signal=signal,
            confidence=confidence,
            confirmed=confirmed,
            reason="; ".join(reasons) if reasons else "All checks passed",
            analysis_time_ms=analysis_time,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        if confirmed:
            self.logger.info(
                f"trade_confirmed symbol={symbol} side={side} score={score} "
                f"time={analysis_time:.1f}ms"
            )
        else:
            self.logger.warning(
                f"trade_rejected symbol={symbol} side={side} score={score} "
                f"reasons={result.reason} time={analysis_time:.1f}ms"
            )

        return result

    @property
    def stats(self) -> dict[str, Any]:
        total = self._confirmation_count + self._rejection_count
        return {
            "total_analyzed": total,
            "confirmed": self._confirmation_count,
            "rejected": self._rejection_count,
            "confirmation_rate": (self._confirmation_count / total * 100) if total > 0 else 0,
            "avg_analysis_time_ms": self._avg_time_ms,
        }
