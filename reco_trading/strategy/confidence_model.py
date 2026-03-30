from __future__ import annotations

from reco_trading.strategy.signal_engine import SignalBundle


class ConfidenceModel:
    """Weighted voting and confidence scoring."""

    def evaluate(self, bundle: SignalBundle, trade_threshold: float = 0.0) -> tuple[str, float, str]:
        explained = self.explain(bundle, trade_threshold=trade_threshold)
        return explained["side"], explained["confidence"], explained["grade"]

    def explain(self, bundle: SignalBundle, trade_threshold: float = 0.0) -> dict[str, object]:
        weighted_votes = {
            "trend": 0.30,
            "momentum": 0.20,
            "volume": 0.08,
            "volatility": 0.08,
            "structure": 0.14,
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

        factor_scores: dict[str, float] = {}
        for name, weight in weighted_votes.items():
            signal = signal_map[name]
            factor_scores[name] = weight if signal == "BUY" else -weight if signal == "SELL" else 0.0

        if buy_score >= sell_score:
            side = "BUY"
            confidence = buy_score
        else:
            side = "SELL"
            confidence = sell_score

        if confidence < max(trade_threshold, 0.0) or (buy_score == sell_score == 0.0):
            side = "HOLD"

        if confidence >= 0.88:
            grade = "EXCEPTIONAL"
        elif confidence >= 0.78:
            grade = "STRONG"
        elif confidence >= 0.62:
            grade = "ACTIONABLE"
        else:
            grade = "WEAK"

        return {
            "side": side,
            "confidence": confidence,
            "grade": grade,
            "buy_score": buy_score,
            "sell_score": sell_score,
            "factor_scores": factor_scores,
            "threshold": max(trade_threshold, 0.0),
        }
