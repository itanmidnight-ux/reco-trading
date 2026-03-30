"""
Meta-Strategy Engine Module.
Manages multiple strategies and dynamic allocation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from reco_trading.advanced_auto_improver.market_regime_detector import MarketRegime

logger = logging.getLogger(__name__)


@dataclass
class StrategyAllocation:
    """Strategy allocation with weight."""
    strategy_id: str
    weight: float
    enabled: bool = True
    performance_score: float = 0.0


@dataclass
class MetaStrategyState:
    """State of meta-strategy."""
    active_strategies: list[StrategyAllocation]
    regime: str
    total_weight: float
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MetaStrategyEngine:
    """Manages multiple strategies with dynamic allocation."""

    def __init__(
        self,
        max_strategies: int = 5,
        min_weight: float = 0.1,
        regime_adjustment: bool = True,
    ):
        self.max_strategies = max_strategies
        self.min_weight = min_weight
        self.regime_adjustment = regime_adjustment
        
        self._strategies: dict[str, StrategyAllocation] = {}
        self._state_history: list[MetaStrategyState] = []
        
        self._regime_weights = {
            MarketRegime.BULL_TREND: {
                "momentum": 0.4,
                "trend": 0.4,
                "mean_reversion": 0.2,
            },
            MarketRegime.BEAR_TREND: {
                "trend": 0.3,
                "mean_reversion": 0.3,
                "momentum": 0.2,
                "contrarian": 0.2,
            },
            MarketRegime.SIDEWAYS: {
                "mean_reversion": 0.4,
                "grid": 0.3,
                "momentum": 0.3,
            },
            MarketRegime.HIGH_VOLATILITY: {
                "mean_reversion": 0.5,
                "contrarian": 0.3,
                "grid": 0.2,
            },
            MarketRegime.LOW_VOLATILITY: {
                "momentum": 0.4,
                "trend": 0.4,
                "mean_reversion": 0.2,
            },
        }

    def register_strategy(
        self,
        strategy_id: str,
        strategy_type: str,
        initial_weight: float = 1.0,
    ) -> None:
        """Register a new strategy."""
        if len(self._strategies) >= self.max_strategies:
            logger.warning(f"Max strategies reached, removing weakest")
            self._remove_weakest_strategy()
        
        self._strategies[strategy_id] = StrategyAllocation(
            strategy_id=strategy_id,
            weight=initial_weight,
            enabled=True,
        )
        
        logger.info(f"Registered strategy: {strategy_id} (type: {strategy_type})")

    def update_strategy_performance(
        self,
        strategy_id: str,
        performance_score: float,
    ) -> None:
        """Update strategy performance score."""
        if strategy_id in self._strategies:
            self._strategies[strategy_id].performance_score = performance_score

    def adjust_for_regime(self, regime: MarketRegime) -> list[StrategyAllocation]:
        """Adjust strategy weights based on market regime."""
        if not self.regime_adjustment:
            return self._get_enabled_strategies()
        
        weights = self._regime_weights.get(regime, self._regime_weights[MarketRegime.SIDEWAYS])
        
        for strategy in self._strategies.values():
            strategy.weight = weights.get(strategy.strategy_id.split("_")[0], 0.5)
        
        self._normalize_weights()
        
        state = MetaStrategyState(
            active_strategies=self._get_enabled_strategies(),
            regime=regime.value,
            total_weight=sum(s.weight for s in self._strategies.values()),
        )
        self._state_history.append(state)
        
        if len(self._state_history) > 1000:
            self._state_history.pop(0)
        
        return state.active_strategies

    def _normalize_weights(self) -> None:
        """Normalize strategy weights to sum to 1.0."""
        total = sum(s.weight for s in self._strategies.values())
        
        if total > 0:
            for strategy in self._strategies.values():
                strategy.weight = strategy.weight / total

    def _get_enabled_strategies(self) -> list[StrategyAllocation]:
        """Get list of enabled strategies."""
        return [s for s in self._strategies.values() if s.enabled]

    def _remove_weakest_strategy(self) -> None:
        """Remove the weakest performing strategy."""
        if not self._strategies:
            return
        
        weakest = min(
            self._strategies.values(),
            key=lambda s: s.performance_score if s.performance_score > 0 else float("inf")
        )
        
        del self._strategies[weakest.strategy_id]
        logger.info(f"Removed weakest strategy: {weakest.strategy_id}")

    def enable_strategy(self, strategy_id: str) -> bool:
        """Enable a strategy."""
        if strategy_id in self._strategies:
            self._strategies[strategy_id].enabled = True
            self._normalize_weights()
            return True
        return False

    def disable_strategy(self, strategy_id: str) -> bool:
        """Disable a strategy."""
        if strategy_id in self._strategies:
            self._strategies[strategy_id].enabled = False
            self._normalize_weights()
            return True
        return False

    def get_best_strategy(self) -> Optional[str]:
        """Get the best performing strategy ID."""
        enabled = self._get_enabled_strategies()
        
        if not enabled:
            return None
        
        best = max(enabled, key=lambda s: s.performance_score)
        return best.strategy_id

    def get_strategy_weights(self) -> dict[str, float]:
        """Get current strategy weights."""
        return {
            s.strategy_id: s.weight
            for s in self._strategies.values()
            if s.enabled
        }

    def get_state(self) -> MetaStrategyState:
        """Get current meta-strategy state."""
        return MetaStrategyState(
            active_strategies=self._get_enabled_strategies(),
            regime="unknown",
            total_weight=sum(s.weight for s in self._strategies.values() if s.enabled),
        )

    def should_use_ensemble(self) -> bool:
        """Determine if ensemble approach should be used."""
        enabled = self._get_enabled_strategies()
        
        if len(enabled) < 2:
            return False
        
        performance_variance = self._calculate_performance_variance(enabled)
        
        return performance_variance < 0.3

    def _calculate_performance_variance(self, strategies: list[StrategyAllocation]) -> float:
        """Calculate variance in strategy performances."""
        if len(strategies) < 2:
            return 0.0
        
        scores = [s.performance_score for s in strategies if s.performance_score > 0]
        
        if len(scores) < 2:
            return 0.0
        
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        
        return variance

    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary."""
        return {
            "strategies": {
                sid: {
                    "weight": s.weight,
                    "enabled": s.enabled,
                    "performance_score": s.performance_score,
                }
                for sid, s in self._strategies.items()
            },
            "max_strategies": self.max_strategies,
            "regime_adjustment": self.regime_adjustment,
            "should_ensemble": self.should_use_ensemble(),
            "best_strategy": self.get_best_strategy(),
            "weights": self.get_strategy_weights(),
        }
