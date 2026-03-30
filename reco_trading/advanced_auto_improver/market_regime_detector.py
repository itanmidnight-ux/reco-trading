"""
Market Regime Detector Module.
Detects and classifies market conditions in real-time.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types."""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


class TrendDirection(Enum):
    """Trend direction."""
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"


@dataclass
class RegimeMetrics:
    """Metrics for regime detection."""
    trend_direction: TrendDirection
    trend_strength: float
    volatility: float
    volatility_percentile: float
    volume_trend: float
    momentum: float
    regime: MarketRegime
    confidence: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MarketRegimeDetector:
    """Detects market regimes using multiple indicators."""

    def __init__(
        self,
        lookback_periods: int = 100,
        volatility_window: int = 20,
        trend_periods: int = 50,
    ):
        self.lookback_periods = lookback_periods
        self.volatility_window = volatility_window
        self.trend_periods = trend_periods
        
        self._regime_history: list[RegimeMetrics] = []
        self._price_history: list[float] = []
        self._volume_history: list[float] = []

    def add_data_point(self, price: float, volume: float, timestamp: Optional[datetime] = None) -> RegimeMetrics:
        """Add a data point and detect regime."""
        self._price_history.append(price)
        self._volume_history.append(volume)
        
        if len(self._price_history) > self.lookback_periods * 2:
            self._price_history.pop(0)
            self._volume_history.pop(0)
        
        if len(self._price_history) < self.trend_periods:
            return self._create_unknown_regime()
        
        return self._detect_regime(price, volume)

    def _detect_regime(self, current_price: float, current_volume: float) -> RegimeMetrics:
        """Detect current market regime."""
        prices = np.array(self._price_history)
        volumes = np.array(self._volume_history)
        
        trend_direction, trend_strength = self._detect_trend(prices)
        volatility = self._calculate_volatility(prices)
        volatility_percentile = self._calculate_volatility_percentile(volatility)
        volume_trend = self._calculate_volume_trend(volumes)
        momentum = self._calculate_momentum(prices)
        
        regime = self._classify_regime(
            trend_direction, trend_strength, volatility, volatility_percentile, momentum
        )
        
        confidence = self._calculate_confidence(trend_strength, volatility, len(prices))
        
        metrics = RegimeMetrics(
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            volatility=volatility,
            volatility_percentile=volatility_percentile,
            volume_trend=volume_trend,
            momentum=momentum,
            regime=regime,
            confidence=confidence,
        )
        
        self._regime_history.append(metrics)
        
        if len(self._regime_history) > 1000:
            self._regime_history.pop(0)
        
        logger.debug(f"Detected regime: {regime.value}, confidence: {confidence:.2f}")
        
        return metrics

    def _detect_trend(self, prices: np.ndarray) -> tuple[TrendDirection, float]:
        """Detect trend direction and strength."""
        if len(prices) < self.trend_periods:
            return TrendDirection.SIDEWAYS, 0.0
        
        recent_prices = prices[-self.trend_periods:]
        
        ma_short = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
        ma_long = np.mean(prices[-self.trend_periods:])
        
        price_change_pct = ((ma_short - ma_long) / ma_long) * 100
        
        if price_change_pct > 2:
            direction = TrendDirection.UP
            strength = min(abs(price_change_pct) / 10, 1.0)
        elif price_change_pct < -2:
            direction = TrendDirection.DOWN
            strength = min(abs(price_change_pct) / 10, 1.0)
        else:
            direction = TrendDirection.SIDEWAYS
            strength = 0.3
        
        return direction, strength

    def _calculate_volatility(self, prices: np.ndarray) -> float:
        """Calculate price volatility (ATR-based)."""
        if len(prices) < self.volatility_window + 1:
            return 0.0
        
        recent_prices = prices[-self.volatility_window - 1:]
        returns = np.diff(recent_prices) / recent_prices[:-1]
        volatility = np.std(returns) * 100 if len(returns) > 0 else 0.0
        
        return volatility

    def _calculate_volatility_percentile(self, current_volatility: float) -> float:
        """Calculate volatility percentile."""
        if len(self._regime_history) < 20:
            return 50.0
        
        volatilities = [m.volatility for m in self._regime_history[-100:]]
        
        if not volatilities:
            return 50.0
        
        sorted_vol = sorted(volatilities)
        rank = sum(1 for v in sorted_vol if v < current_volatility)
        
        return (rank / len(sorted_vol)) * 100

    def _calculate_volume_trend(self, volumes: np.ndarray) -> float:
        """Calculate volume trend."""
        if len(volumes) < 20:
            return 0.0
        
        recent_avg = np.mean(volumes[-10:])
        older_avg = np.mean(volumes[-20:-10]) if len(volumes) >= 20 else recent_avg
        
        if older_avg == 0:
            return 0.0
        
        return ((recent_avg - older_avg) / older_avg) * 100

    def _calculate_momentum(self, prices: np.ndarray) -> float:
        """Calculate price momentum."""
        if len(prices) < 14:
            return 0.0
        
        rsi = self._calculate_rsi(prices)
        
        return (rsi - 50) / 50

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices[-period - 1:])
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def _classify_regime(
        self,
        trend_direction: TrendDirection,
        trend_strength: float,
        volatility: float,
        volatility_percentile: float,
        momentum: float,
    ) -> MarketRegime:
        """Classify market regime."""
        if volatility_percentile > 80:
            return MarketRegime.HIGH_VOLATILITY
        
        if volatility_percentile < 20:
            return MarketRegime.LOW_VOLATILITY
        
        if trend_direction == TrendDirection.UP and trend_strength > 0.5:
            return MarketRegime.BULL_TREND
        
        if trend_direction == TrendDirection.DOWN and trend_strength > 0.5:
            return MarketRegime.BEAR_TREND
        
        return MarketRegime.SIDEWAYS

    def _calculate_confidence(self, trend_strength: float, volatility: float, data_points: int) -> float:
        """Calculate detection confidence."""
        confidence = trend_strength * 0.4
        
        if volatility < 5:
            confidence += 0.3
        elif volatility < 10:
            confidence += 0.2
        else:
            confidence += 0.1
        
        data_confidence = min(data_points / self.lookback_periods, 1.0) * 0.3
        
        confidence += data_confidence
        
        return min(confidence, 1.0)

    def _create_unknown_regime(self) -> RegimeMetrics:
        """Create unknown regime for insufficient data."""
        return RegimeMetrics(
            trend_direction=TrendDirection.SIDEWAYS,
            trend_strength=0.0,
            volatility=0.0,
            volatility_percentile=50.0,
            volume_trend=0.0,
            momentum=0.0,
            regime=MarketRegime.UNKNOWN,
            confidence=0.0,
        )

    def get_current_regime(self) -> Optional[RegimeMetrics]:
        """Get current regime."""
        if not self._regime_history:
            return None
        return self._regime_history[-1]

    def get_regime_history(self, limit: int = 100) -> list[RegimeMetrics]:
        """Get regime history."""
        return self._regime_history[-limit:]

    def should_trade(self) -> tuple[bool, str]:
        """Determine if trading is recommended in current regime."""
        regime = self.get_current_regime()
        
        if not regime:
            return False, "Insufficient data"
        
        if regime.regime == MarketRegime.UNKNOWN:
            return False, "Unknown regime"
        
        if regime.regime == MarketRegime.HIGH_VOLATILITY:
            return False, "High volatility - risky"
        
        if regime.regime == MarketRegime.LOW_VOLATILITY:
            return True, "Low volatility - conservative trading"
        
        if regime.confidence < 0.5:
            return False, "Low confidence in regime detection"
        
        return True, f"Trading allowed in {regime.regime.value}"

    def get_recommended_position_size(self, base_size: float) -> float:
        """Get recommended position size based on regime."""
        regime = self.get_current_regime()
        
        if not regime:
            return base_size * 0.5
        
        if regime.regime == MarketRegime.HIGH_VOLATILITY:
            return base_size * 0.3
        
        if regime.regime == MarketRegime.LOW_VOLATILITY:
            return base_size * 0.8
        
        if regime.regime == MarketRegime.BEAR_TREND:
            return base_size * 0.5
        
        return base_size * regime.confidence

    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary."""
        regime = self.get_current_regime()
        
        if not regime:
            return {"status": "no_data"}
        
        return {
            "regime": regime.regime.value,
            "trend_direction": regime.trend_direction.value,
            "trend_strength": regime.trend_strength,
            "volatility": regime.volatility,
            "volatility_percentile": regime.volatility_percentile,
            "momentum": regime.momentum,
            "confidence": regime.confidence,
            "timestamp": regime.timestamp.isoformat(),
            "should_trade": self.should_trade()[0],
        }
