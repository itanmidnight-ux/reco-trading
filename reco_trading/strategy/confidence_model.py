from __future__ import annotations

from reco_trading.strategy.signal_engine import SignalBundle


class ConfidenceModel:
    """Signal voting and confidence calculation."""

    def evaluate(self, bundle: SignalBundle) -> tuple[str, float]:
        votes = [bundle.trend, bundle.momentum, bundle.volume, bundle.volatility, bundle.structure]
        buy_votes = sum(1 for vote in votes if vote == "BUY")
        sell_votes = sum(1 for vote in votes if vote == "SELL")
        total = len(votes)

        if buy_votes >= sell_votes:
            side = "BUY"
            confidence = buy_votes / total
        else:
            side = "SELL"
            confidence = sell_votes / total
        return side, confidence
