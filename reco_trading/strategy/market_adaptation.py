"""
Dynamic Market Adaptation Module.

Automatically adjusts trading parameters based on market conditions:
- Detects sideways/ranging vs trending markets
- Relaxes filters for stable markets while maintaining safety
- Switches to scalping mode when appropriate
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MarketState:
    """Current market state assessment."""
    regime: str = "UNKNOWN"  # TRENDING, RANGING, VOLATILE, STABLE
    volatility_percentile: float = 50.0  # 0-100
    momentum_strength: float = 0.0  # -1 to 1
    trending: bool = False
    sideways: bool = False
    volatile: bool = False
    optimal_strategy: str = "DEFAULT"  # TREND_FOLLOW, SCALP, SWING, AVOID
    confidence_modifier: float = 0.0
    adx_adjustment: float = 0.0
    rsi_adjustment: float = 0.0
    volume_adjustment: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    signal_count_1h: int = 0
    trades_count_1h: int = 0


class MarketAdaptation:
    """
    Adapts trading parameters based on market conditions.
    
    Strategies:
    - TRENDING: Normal trend-following (default)
    - RANGING: Scalping with tighter stops, relaxed filters
    - VOLATILE: Reduced size, wider stops
    - STABLE: Higher threshold OR switch pairs
    """
    
    # Volatility thresholds
    LOW_VOLATILITY_ATR_RATIO = 0.015  # ATR/Price below this = stable
    HIGH_VOLATILITY_ATR_RATIO = 0.05   # ATR/Price above this = volatile
    
    # ADX thresholds
    TRENDING_ADX_MIN = 25
    RANGING_ADX_MAX = 20
    
    # RSI ranges for different markets
    TRENDING_RSI_BUY_MIN = 58
    TRENDING_RSI_SELL_MAX = 42
    RANGING_RSI_BUY_MIN = 50  # More aggressive in ranging
    RANGING_RSI_SELL_MAX = 50
    
    # Confidence adjustments
    TRENDING_CONFIDENCE_MIN = 0.50
    RANGING_CONFIDENCE_MIN = 0.38  # Lower in sideways
    
    def __init__(self) -> None:
        self.state = MarketState()
        self._atr_history: list[float] = []
        self._adx_history: list[float] = []
        self._signal_history: list[datetime] = []
        self._trade_history: list[datetime] = []
        
        # Adaptive thresholds
        self._base_adx_threshold = 18.0
        self._base_confidence = 0.55
        self._base_rsi_buy = 58
        self._base_rsi_sell = 42
        
        # Current adjusted values
        self.adx_threshold = self._base_adx_threshold
        self.confidence_min = self._base_confidence
        self.rsi_buy_threshold = self._base_rsi_buy
        self.rsi_sell_threshold = self._base_rsi_sell
        self.volume_threshold = 0.80
        
        # Mode
        self.scalping_mode = False
        self.pair_switch_recommended = False
        
    def update(self, price: float, atr: float, adx: float, 
               rsi: float, volume_ratio: float,
               signal_confidence: float) -> MarketState:
        """Update market state and adapt parameters."""
        now = datetime.now(timezone.utc)
        
        # Track history
        if atr > 0:
            self._atr_history.append(atr / max(price, 1e-9))
            self._atr_history = self._atr_history[-100:]
        
        self._adx_history.append(adx)
        self._adx_history = self._adx_history[-100:]
        
        # Calculate ATR percentile
        atr_ratio = atr / max(price, 1e-9)
        if self._atr_history:
            sorted_atr = sorted(self._atr_history)
            atr_percentile = sum(1 for x in sorted_atr if x <= atr_ratio) / len(sorted_atr) * 100
        else:
            atr_percentile = 50.0
        
        # Determine regime
        if atr_ratio < self.LOW_VOLATILITY_ATR_RATIO:
            self.state.regime = "STABLE"
            self.state.sideways = True
            self.state.trending = False
            self.state.volatile = False
            self.state.optimal_strategy = "SCALP"
        elif atr_ratio > self.HIGH_VOLATILITY_ATR_RATIO:
            self.state.regime = "VOLATILE"
            self.state.volatile = True
            self.state.trending = False
            self.state.sideways = False
            self.state.optimal_strategy = "TREND_FOLLOW"
        elif adx < self.RANGING_ADX_MAX:
            self.state.regime = "RANGING"
            self.state.sideways = True
            self.state.trending = False
            self.state.volatile = False
            self.state.optimal_strategy = "SCALP"
        elif adx > self.TRENDING_ADX_MIN:
            self.state.regime = "TRENDING"
            self.state.trending = True
            self.state.sideways = False
            self.state.volatile = False
            self.state.optimal_strategy = "TREND_FOLLOW"
        else:
            self.state.regime = "MODERATE"
            self.state.optimal_strategy = "DEFAULT"
        
        self.state.volatility_percentile = atr_percentile
        self.state.momentum_strength = (rsi - 50) / 50  # -1 to 1
        self.state.last_updated = now
        
        # Adapt thresholds based on regime
        self._adapt_thresholds()
        
        # Clean old history
        hour_ago = now - timedelta(hours=1)
        self._signal_history = [t for t in self._signal_history if t > hour_ago]
        self._trade_history = [t for t in self._trade_history if t > hour_ago]
        
        self.state.signal_count_1h = len(self._signal_history)
        self.state.trades_count_1h = len(self._trade_history)
        
        # Recommend pair switch if too few signals
        if self.state.regime in ("STABLE", "RANGING"):
            if self.state.signal_count_1h < 3:
                self.pair_switch_recommended = True
                self.state.optimal_strategy = "SWITCH_PAIR"
            else:
                self.pair_switch_recommended = False
        else:
            self.pair_switch_recommended = False
        
        return self.state
    
    def _adapt_thresholds(self) -> None:
        """Adapt thresholds based on current regime."""
        if self.state.regime == "STABLE":
            # Very stable market - significant relaxation or switch pairs
            self.adx_threshold = max(12, self._base_adx_threshold - 6)
            self.confidence_min = max(0.32, self._base_confidence - 0.18)
            self.rsi_buy_threshold = max(48, self._base_rsi_buy - 10)
            self.rsi_sell_threshold = min(52, self._base_rsi_sell + 10)
            self.volume_threshold = 0.60
            self.scalping_mode = True
            
        elif self.state.regime == "RANGING":
            # Sideways market - moderate relaxation
            self.adx_threshold = max(14, self._base_adx_threshold - 4)
            self.confidence_min = max(0.38, self._base_confidence - 0.12)
            self.rsi_buy_threshold = max(52, self._base_rsi_buy - 6)
            self.rsi_sell_threshold = min(48, self._base_rsi_sell + 6)
            self.volume_threshold = 0.70
            self.scalping_mode = True
            
        elif self.state.regime == "VOLATILE":
            # High volatility - tighter filters
            self.adx_threshold = self._base_adx_threshold + 5
            self.confidence_min = min(0.65, self._base_confidence + 0.10)
            self.rsi_buy_threshold = 60
            self.rsi_sell_threshold = 40
            self.volume_threshold = 0.90
            self.scalping_mode = False
            
        else:  # TRENDING, MODERATE, DEFAULT
            # Normal conditions - use base thresholds
            self.adx_threshold = self._base_adx_threshold
            self.confidence_min = self._base_confidence
            self.rsi_buy_threshold = self._base_rsi_buy
            self.rsi_sell_threshold = self._base_rsi_sell
            self.volume_threshold = 0.80
            self.scalping_mode = False
    
    def record_signal(self) -> None:
        """Record that a signal was generated."""
        self._signal_history.append(datetime.now(timezone.utc))
    
    def record_trade(self) -> None:
        """Record that a trade was executed."""
        self._trade_history.append(datetime.now(timezone.utc))
    
    def get_adjusted_params(self) -> dict[str, Any]:
        """Get current adjusted parameters."""
        return {
            "adx_threshold": self.adx_threshold,
            "confidence_min": self.confidence_min,
            "rsi_buy_threshold": self.rsi_buy_threshold,
            "rsi_sell_threshold": self.rsi_sell_threshold,
            "volume_threshold": self.volume_threshold,
            "scalping_mode": self.scalping_mode,
            "regime": self.state.regime,
            "optimal_strategy": self.state.optimal_strategy,
            "pair_switch_recommended": self.pair_switch_recommended,
            "signal_count_1h": self.state.signal_count_1h,
            "trades_count_1h": self.state.trades_count_1h,
        }
    
    def should_switch_pair(self) -> bool:
        """Check if should switch to a different pair."""
        if self.state.regime == "STABLE":
            return self.state.signal_count_1h < 2
        elif self.state.regime == "RANGING":
            return self.state.signal_count_1h < 4
        return False
    
    def get_stop_take_multiplier(self) -> tuple[float, float]:
        """Get ATR multipliers for stop loss and take profit.
        
        Returns (stop_multiplier, take_multiplier)
        """
        if self.scalping_mode:
            # Tighter stops for scalping
            return (1.2, 1.5)  # Smaller stops, smaller targets
        elif self.state.regime == "VOLATILE":
            # Wider stops for volatility
            return (2.5, 4.0)
        else:
            # Normal
            return (2.0, 3.0)
    
    def get_position_size_multiplier(self) -> float:
        """Get position size multiplier based on conditions."""
        if self.state.regime == "VOLATILE":
            return 0.5  # Smaller positions in volatility
        elif self.state.regime == "STABLE":
            return 0.75  # Slightly smaller in stable
        elif self.scalping_mode:
            return 0.6  # Smaller for scalping
        else:
            return 1.0  # Normal size


# Top volatile trading pairs by 24h volume and volatility
RECOMMIGNED_PAIRS = [
    # Tier 1: Major pairs with good volatility
    "BTC/USDT", "ETH/USDT", "SOL/USDT",
    # Tier 2: High volatility altcoins
    "DOGE/USDT", "XRP/USDT", "AVAX/USDT",
    # Tier 3: Meme coins (extreme volatility)
    "PEPE/USDT", "WIF/USDT", "BONK/USDT",
    # Tier 4: DeFi tokens
    "LINK/USDT", "UNI/USDT", "AAVE/USDT",
]

PAIR_TIERS = {
    1: ["BTC/USDT", "ETH/USDT", "SOL/USDT"],  # Most reliable
    2: ["DOGE/USDT", "XRP/USDT", "AVAX/USDT"],  # High volatility
    3: ["PEPE/USDT", "WIF/USDT"],  # Extreme volatility - use with caution
}