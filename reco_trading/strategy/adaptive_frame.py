"""
Adaptive Frame Engine for Multi-Timeframe Analysis.

Automatically switches between timeframes based on market conditions:
- Uses 1m/5m for scalping in ranging markets
- Uses 5m/15m for normal conditions
- Uses 15m/1h for volatile/trending markets
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any
from enum import Enum

logger = logging.getLogger(__name__)


class FrameMode(Enum):
    """Timeframe modes for different market conditions."""
    SCALP = "scalp"      # 1m primary, 5m confirmation - for ranging/stable
    DEFAULT = "default"  # 5m primary, 15m confirmation - moderate conditions
    TREND = "trend"      # 15m primary, 1h confirmation - trending
    AVOID = "avoid"      # No trading - extreme conditions


@dataclass
class FrameConfig:
    """Configuration for a timeframe combination."""
    primary: str = "5m"
    confirmation: str = "15m"
    primary_lookback: int = 100
    confirmation_lookback: int = 50
    min_weight: float = 0.3
    max_weight: float = 1.0
    name: str = "DEFAULT"


@dataclass
class FrameState:
    """Current state of adaptive frame engine."""
    mode: FrameMode = FrameMode.DEFAULT
    current_config: FrameConfig = field(default_factory=FrameConfig)
    primary_weight: float = 0.5
    confirmation_weight: float = 0.5
    last_mode_change: datetime = field(default_factory=datetime.now)
    mode_stability_score: float = 1.0
    recommended_action: str = "TRADE"
    reason: str = "Normal conditions"


class AdaptiveFrameEngine:
    """
    Manages timeframe selection based on market conditions.
    
    Mode Logic:
    - SCALP: Low volatility, ranging (ADX < 20), stable markets
      - Use: 1m primary, 5m confirmation
      - More signals, smaller targets
      
    - DEFAULT: Moderate volatility, normal conditions
      - Use: 5m primary, 15m confirmation
      - Standard trend following
      
    - TREND: High ADX (> 25), strong trend or high volatility
      - Use: 15m primary, 1h confirmation
      - Larger targets, fewer signals
      
    - AVOID: Extreme conditions
      - No trading recommended
    """
    
    # Mode thresholds
    SCALP_ADX_MAX = 18
    SCALP_VOLATILITY_MAX = 0.015  # ATR/Price
    SCALP_SIGNALS_MAX = 6  # Max signals per hour
    
    TREND_ADX_MIN = 25
    TREND_VOLATILITY_MIN = 0.035
    
    AVOID_VOLATILITY_MIN = 0.06
    AVOID_DRAWDOWN_MAX = -0.10  # 10% drawdown
    
    # Minimum time before mode switch
    MIN_MODE_DURATION_MINUTES = 15
    
    # Frame configurations
    FRAME_CONFIGS = {
        FrameMode.SCALP: FrameConfig(
            primary="1m",
            confirmation="5m",
            primary_lookback=150,
            confirmation_lookback=80,
            min_weight=0.4,
            max_weight=1.0,
            name="SCALP"
        ),
        FrameMode.DEFAULT: FrameConfig(
            primary="5m",
            confirmation="15m",
            primary_lookback=100,
            confirmation_lookback=50,
            min_weight=0.3,
            max_weight=1.0,
            name="DEFAULT"
        ),
        FrameMode.TREND: FrameConfig(
            primary="15m",
            confirmation="1h",
            primary_lookback=80,
            confirmation_lookback=30,
            min_weight=0.25,
            max_weight=0.8,
            name="TREND"
        ),
    }
    
    def __init__(self) -> None:
        self.state = FrameState()
        self._mode_history: list[tuple[datetime, FrameMode]] = []
        self._volatility_history: list[float] = []
        self._adx_history: list[float] = []
        self._signal_history: list[datetime] = []
        
        # Confluence weights
        self.primary_weight = 0.6
        self.confirmation_weight = 0.4
        
        # Mode change cooldown
        self._last_mode_check: datetime | None = None
        self._consecutive_same_mode: int = 0
        
    def assess_market(
        self,
        price: float,
        atr: float,
        adx: float,
        rsi: float,
        volume_ratio: float,
        regime: str,
        daily_pnl: float = 0.0,
        drawdown_pct: float = 0.0,
        signals_last_hour: int = 0,
    ) -> FrameState:
        """
        Assess market and determine optimal timeframe mode.
        
        Returns updated FrameState with recommendations.
        """
        now = datetime.now(timezone.utc)
        
        # Track history
        if atr > 0 and price > 0:
            vol_ratio = atr / price
            self._volatility_history.append(vol_ratio)
            self._volatility_history = self._volatility_history[-50:]
        
        self._adx_history.append(adx)
        self._adx_history = self._adx_history[-50:]
        
        # Determine recommended mode
        recommended_mode = self._determine_mode(
            atr_ratio=atr / max(price, 1e-9),
            adx=adx,
            regime=regime,
            drawdown_pct=drawdown_pct,
            signals_last_hour=signals_last_hour,
        )
        
        # Check if we should switch modes (with cooldown)
        time_since_change = (now - self.state.last_mode_change).total_seconds() / 60
        can_switch = (
            time_since_change >= self.MIN_MODE_DURATION_MINUTES or
            recommended_mode == FrameMode.AVOID
        )
        
        if can_switch and recommended_mode != self.state.mode:
            self._apply_mode_switch(recommended_mode, now)
        else:
            # Update stability score
            if recommended_mode == self.state.mode:
                self._consecutive_same_mode += 1
                self.state.mode_stability_score = min(1.0, self.state.mode_stability_score + 0.05)
            else:
                self._consecutive_same_mode = 0
                self.state.mode_stability_score = max(0.5, self.state.mode_stability_score - 0.1)
        
        # Update weights based on regime
        self._update_weights(regime, adx)
        
        # Determine action recommendation
        self._determine_action(drawdown_pct, signals_last_hour)
        
        # Clean old history
        hour_ago = now - timedelta(hours=1)
        self._signal_history = [t for t in self._signal_history if t > hour_ago]
        
        self._last_mode_check = now
        
        return self.state
    
    def _determine_mode(
        self,
        atr_ratio: float,
        adx: float,
        regime: str,
        drawdown_pct: float,
        signals_last_hour: int,
    ) -> FrameMode:
        """Determine the optimal timeframe mode."""
        
        # Avoid mode for extreme conditions
        if abs(drawdown_pct) > abs(self.AVOID_DRAWDOWN_MAX):
            return FrameMode.AVOID
        
        if atr_ratio > self.AVOID_VOLATILITY_MIN:
            return FrameMode.AVOID
        
        # Scalp mode for ranging/stable with low volatility
        if (
            atr_ratio < self.SCALP_VOLATILITY_MAX and
            adx < self.SCALP_ADX_MAX and
            regime in ("RANGING", "STABLE") and
            signals_last_hour < self.SCALP_SIGNALS_MAX
        ):
            return FrameMode.SCALP
        
        # Trend mode for strong trends or high volatility
        if (
            adx > self.TREND_ADX_MIN or
            atr_ratio > self.TREND_VOLATILITY_MIN or
            regime in ("TRENDING", "VOLATILE")
        ):
            return FrameMode.TREND
        
        # Default for moderate conditions
        return FrameMode.DEFAULT
    
    def _apply_mode_switch(self, new_mode: FrameMode, now: datetime) -> None:
        """Apply a mode switch."""
        old_mode = self.state.mode
        self.state.mode = new_mode
        
        if new_mode in self.FRAME_CONFIGS:
            self.state.current_config = self.FRAME_CONFIGS[new_mode]
        
        self.state.last_mode_change = now
        self._mode_history.append((now, new_mode))
        self._mode_history = self._mode_history[-20:]
        self._consecutive_same_mode = 0
        self.state.mode_stability_score = 0.7
        
        logger.info(
            f"Frame mode switched: {old_mode.value} -> {new_mode.value} "
            f"(primary={self.state.current_config.primary}, "
            f"confirmation={self.state.current_config.confirmation})"
        )
    
    def _update_weights(self, regime: str, adx: float) -> None:
        """Update confluence weights based on conditions."""
        if regime == "TRENDING" and adx > 25:
            # Higher timeframe more important in trend
            self.state.primary_weight = 0.55
            self.state.confirmation_weight = 0.45
        elif regime in ("RANGING", "STABLE"):
            # Primary timeframe more important for scalping
            self.state.primary_weight = 0.65
            self.state.confirmation_weight = 0.35
        else:
            # Balanced
            self.state.primary_weight = 0.6
            self.state.confirmation_weight = 0.4
        
        self.primary_weight = self.state.primary_weight
        self.confirmation_weight = self.state.confirmation_weight
    
    def _determine_action(self, drawdown_pct: float, signals_last_hour: int) -> None:
        """Determine recommended action."""
        if self.state.mode == FrameMode.AVOID:
            self.state.recommended_action = "PAUSE"
            self.state.reason = "Extreme conditions detected"
            return
        
        if abs(drawdown_pct) > 0.05:
            self.state.recommended_action = "REDUCE"
            self.state.reason = "Drawdown limits approaching"
            return
        
        if signals_last_hour > 10 and self.state.mode == FrameMode.SCALP:
            self.state.recommended_action = "SLOW"
            self.state.reason = "Too many signals in scalp mode"
            return
        
        self.state.recommended_action = "TRADE"
        self.state.reason = "Normal conditions"
    
    def record_signal(self) -> None:
        """Record that a signal was generated."""
        self._signal_history.append(datetime.now(timezone.utc))
    
    def get_timeframes(self) -> tuple[str, str]:
        """Get current timeframes (primary, confirmation)."""
        config = self.state.current_config
        return config.primary, config.confirmation
    
    def get_weights(self) -> tuple[float, float]:
        """Get current confluence weights (primary, confirmation)."""
        return self.state.primary_weight, self.state.confirmation_weight
    
    def get_lookback(self) -> tuple[int, int]:
        """Get lookback periods (primary, confirmation)."""
        config = self.state.current_config
        return config.primary_lookback, config.confirmation_lookback
    
    def should_trade(self) -> tuple[bool, str]:
        """
        Check if trading is recommended.
        
        Returns (should_trade, reason).
        """
        if self.state.mode == FrameMode.AVOID:
            return False, self.state.reason
        
        if self.state.recommended_action == "PAUSE":
            return False, self.state.reason
        
        return True, self.state.reason
    
    def get_mode_info(self) -> dict[str, Any]:
        """Get complete mode information."""
        config = self.state.current_config
        return {
            "mode": self.state.mode.value,
            "primary_timeframe": config.primary,
            "confirmation_timeframe": config.confirmation,
            "primary_weight": self.state.primary_weight,
            "confirmation_weight": self.state.confirmation_weight,
            "primary_lookback": config.primary_lookback,
            "confirmation_lookback": config.confirmation_lookback,
            "mode_stability": self.state.mode_stability_score,
            "recommended_action": self.state.recommended_action,
            "reason": self.state.reason,
            "time_since_change": (datetime.now(timezone.utc) - self.state.last_mode_change).total_seconds() / 60,
        }
    
    def get_confluence_score(
        self,
        primary_signal: str,
        confirmation_signal: str,
        primary_confidence: float,
        confirmation_confidence: float,
    ) -> float:
        """
        Calculate weighted confluence score.
        
        Returns score from 0 to 1.
        """
        # Direction agreement
        if primary_signal != confirmation_signal:
            # Signals disagree - reduce score
            if primary_signal == "HOLD" or confirmation_signal == "HOLD":
                return 0.0
            return 0.2  # Opposing signals
        
        # Same direction - weighted confidence
        weighted_confidence = (
            primary_confidence * self.state.primary_weight +
            confirmation_confidence * self.state.confirmation_weight
        )
        
        # Bonus for strong agreement
        if primary_confidence > 0.75 and confirmation_confidence > 0.75:
            weighted_confidence = min(1.0, weighted_confidence * 1.1)
        
        return weighted_confidence


__all__ = [
    "AdaptiveFrameEngine",
    "FrameMode",
    "FrameConfig",
    "FrameState",
]