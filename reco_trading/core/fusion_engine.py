from __future__ import annotations

from reco_trading.core.signal_fusion_engine import SignalFusionEngine, SignalObservation


class FusionEngine:
    """Compat layer sobre el nuevo SignalFusionEngine."""

    def __init__(self) -> None:
        self.engine = SignalFusionEngine(model_names=["momentum", "mean_reversion"])

    def decide(self, momentum_up_prob: float, reversion_prob: float) -> str:
        result = self.engine.fuse(
            [
                SignalObservation(name="momentum", score=2.0 * momentum_up_prob - 1.0, confidence=abs(momentum_up_prob - 0.5) * 2),
                SignalObservation(name="mean_reversion", score=2.0 * reversion_prob - 1.0, confidence=abs(reversion_prob - 0.5) * 2),
            ]
        )
        if result.calibrated_probability >= 0.57:
            return "BUY"
        if result.calibrated_probability <= 0.43:
            return "SELL"
        return "HOLD"
