from __future__ import annotations

from reco_trading.strategy.signal_engine import SignalBundle


class ConfidenceModel:
    """Weighted voting and confidence scoring with conflict detection."""

    # Minimum confidence required to generate a trade signal
    MIN_CONFIDENCE_THRESHOLD = 0.35
    
    # Weights for each signal factor
    WEIGHTS = {
        "trend": 0.30,
        "momentum": 0.20,
        "volume": 0.08,
        "volatility": 0.08,
        "structure": 0.14,
        "order_flow": 0.20,
    }

    def evaluate(self, bundle: SignalBundle, trade_threshold: float = 0.0) -> tuple[str, float, str]:
        explained = self.explain(bundle, trade_threshold=trade_threshold)
        return explained["side"], explained["confidence"], explained["grade"]

    def explain(self, bundle: SignalBundle, trade_threshold: float = 0.0) -> dict[str, object]:
        weighted_votes = self.WEIGHTS.copy()
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
        
        # Count signal agreement - number of factors agreeing
        buy_count = sum(1 for name in weighted_votes if signal_map[name] == "BUY")
        sell_count = sum(1 for name in weighted_votes if signal_map[name] == "SELL")
        neutral_count = sum(1 for name in weighted_votes if signal_map[name] == "NEUTRAL")

        factor_scores: dict[str, float] = {}
        for name, weight in weighted_votes.items():
            signal = signal_map[name]
            factor_scores[name] = weight if signal == "BUY" else -weight if signal == "SELL" else 0.0

        # Calculate conflict penalty
        # Higher conflict = more uncertainty = lower confidence
        conflict_score = min(buy_score, sell_score)
        conflict_penalty = conflict_score * 0.5  # 50% penalty for conflicting signals
        
        # Net confidence after conflict penalty
        if buy_score >= sell_score:
            side = "BUY"
            raw_confidence = buy_score
            confidence = max(0.0, buy_score - conflict_penalty)
        else:
            side = "SELL"
            raw_confidence = sell_score
            confidence = max(0.0, sell_score - conflict_penalty)

        # Apply minimum confidence threshold
        effective_threshold = max(trade_threshold, self.MIN_CONFIDENCE_THRESHOLD)
        
        # Require minimum signal agreement (at least 2 factors must agree)
        min_factors_required = 2
        factors_agreeing = buy_count if side == "BUY" else sell_count
        
        if confidence < effective_threshold:
            side = "HOLD"
        elif factors_agreeing < min_factors_required:
            # Not enough agreement - weak signal
            side = "HOLD"
            confidence = raw_confidence * 0.5  # Reduce confidence significantly
        elif buy_score == sell_score == 0.0:
            side = "HOLD"

        # Improved grade thresholds - more conservative
        if confidence >= 0.85:
            grade = "EXCEPTIONAL"
        elif confidence >= 0.75:
            grade = "STRONG"
        elif confidence >= 0.60:
            grade = "ACTIONABLE"
        elif confidence >= 0.45:
            grade = "WEAK"
        else:
            grade = "VERY_WEAK"

        # Signal quality indicators
        signal_quality = {
            "agreement_ratio": factors_agreeing / len(weighted_votes),
            "conflict_level": conflict_score,
            "neutral_ratio": neutral_count / len(weighted_votes),
        }

        return {
            "side": side,
            "confidence": confidence,
            "grade": grade,
            "buy_score": buy_score,
            "sell_score": sell_score,
            "factor_scores": factor_scores,
            "threshold": effective_threshold,
            "conflict_penalty": conflict_penalty,
            "signal_quality": signal_quality,
        }
