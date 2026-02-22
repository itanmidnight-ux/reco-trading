from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reco_trading.core.market_state import MarketState


@dataclass(frozen=True, slots=True)
class Decision:
    action: str
    confidence: float
    reason: str
    scores: dict[str, float]


class DecisionEngine:
    """Single source of truth for action, confidence and scores."""

    def __init__(self, min_edge: float = 0.0005) -> None:
        self._min_edge = float(min_edge)
        self._market_state: MarketState | None = None
        self.last_confidence: float = 0.0
        self.last_scores: dict[str, float] = {}
        self.last_reason: str = 'booting'

    def update_context(self, market_state: MarketState) -> None:
        self._market_state = market_state

    def decide(self) -> Decision:
        if self._market_state is None:
            decision = Decision('HOLD', 0.0, 'missing_market_state', {'global': 0.0})
            self._store(decision)
            return decision

        state = self._market_state
        confidence = float(np.clip((state.expected_edge + 1.0) / 2.0, 0.0, 1.0))
        scores = {
            'edge': float(state.expected_edge),
            'friction': float(state.friction_cost),
            'regime_tradable': 1.0 if state.regime.tradable else 0.0,
            'global': confidence,
        }

        if not state.regime.tradable:
            decision = Decision('HOLD', confidence, f'regime_blocked:{state.regime.name}', scores)
        elif state.expected_edge <= state.friction_cost:
            decision = Decision('HOLD', confidence, 'edge_below_friction', scores)
        elif abs(state.expected_edge) < self._min_edge:
            decision = Decision('HOLD', confidence, 'edge_below_threshold', scores)
        elif state.expected_edge > 0:
            decision = Decision('BUY', confidence, 'positive_statistical_edge', scores)
        else:
            decision = Decision('SELL', confidence, 'negative_statistical_edge', scores)

        self._store(decision)
        return decision

    def _store(self, decision: Decision) -> None:
        self.last_confidence = decision.confidence
        self.last_scores = decision.scores
        self.last_reason = decision.reason
