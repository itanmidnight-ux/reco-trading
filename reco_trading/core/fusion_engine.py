from __future__ import annotations


class FusionEngine:
    def decide(self, momentum_up_prob: float, reversion_prob: float) -> str:
        momentum_signal = 'BUY' if momentum_up_prob >= 0.55 else 'SELL' if momentum_up_prob <= 0.45 else 'HOLD'
        reversion_signal = 'BUY' if reversion_prob >= 0.55 else 'SELL' if reversion_prob <= 0.45 else 'HOLD'

        if 'HOLD' in {momentum_signal, reversion_signal}:
            return 'HOLD'
        if momentum_signal == reversion_signal:
            return momentum_signal
        return 'HOLD'
