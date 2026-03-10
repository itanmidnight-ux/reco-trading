from __future__ import annotations

from reco_trading.strategy.signal_engine import SignalBundle


class ConfidenceModel:
    """Weighted voting and confidence scoring."""

    def evaluate(self, bundle: SignalBundle, trade_threshold: float) -> tuple[str, float, str]:
        weighted_votes = {
            "trend": 0.25,
            "momentum": 0.15,
            "volume": 0.10,
            "volatility": 0.15,
            "structure": 0.15,
            "order_flow": 0.20,
        }
        signal_map = {
            "trend": bundle.trend,
            "momentum": bundle.momentum,
            "volume": bundle.volume,
            "volatility": bundle.volatility,
            "structure": bundle.structure,
            "order_flow": bundle.order_flow,
        }

        buy_score = sum(weight for name, weight in weighted_votes.items() if signal_map[name] == "BUY")
        sell_score = sum(weight for name, weight in weighted_votes.items() if signal_map[name] == "SELL")

        if buy_score >= sell_score:
            side = "BUY"
            confidence = buy_score
        else:
            side = "SELL"
            confidence = sell_score

        if confidence < trade_threshold:
            side = "HOLD"

        if confidence >= 0.90:
            grade = "EXCEPTIONAL"
        elif confidence >= 0.85:
            grade = "STRONG"
        elif confidence >= 0.75:
            grade = "ACTIONABLE"
        else:
            grade = "WEAK"

        return side, confidence, grade
