from __future__ import annotations

from reco_trading.core.signal_fusion import SignalFusionEngine


class FusionEngine:
    """Compat layer sobre el SignalFusionEngine institucional."""

    def __init__(self) -> None:
        self.engine = SignalFusionEngine()

    def decide(self, momentum_up_prob: float, reversion_prob: float, regime: str = 'trend', volatility: float = 0.01) -> str:
        fused_probability = self.engine.fuse(
            signals={
                'momentum': 2.0 * momentum_up_prob - 1.0,
                'mean_reversion': 2.0 * reversion_prob - 1.0,
            },
            regime=regime,
            volatility=volatility,
            confidences={
                'momentum': abs(momentum_up_prob - 0.5) * 2,
                'mean_reversion': abs(reversion_prob - 0.5) * 2,
            },
        )
        if fused_probability >= 0.57:
            return 'BUY'
        if fused_probability <= 0.43:
            return 'SELL'
        return 'HOLD'
