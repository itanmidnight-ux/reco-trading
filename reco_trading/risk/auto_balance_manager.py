"""
Auto Balance Detection and Intelligent Position Sizing.

This module automatically:
1. Detects available capital from exchange balance
2. Calculates optimal position size based on signal strength
3. Adapts to market conditions and account size
4. Protects capital during drawdowns
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CapitalState:
    """Current state of trading capital."""
    total_balance: float = 0.0
    available_balance: float = 0.0
    unrealized_pnl: float = 0.0
    margin_used: float = 0.0
    free_collateral: float = 0.0
    peak_balance: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown_seen: float = 0.0
    trades_today: int = 0
    pnl_today: float = 0.0
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PositionSizeRecommendation:
    """Recommended position sizing."""
    quantity: float
    notional_value: float
    risk_amount: float
    risk_percentage: float
    confidence_multiplier: float
    drawdown_multiplier: float
    volatility_multiplier: float
    streak_multiplier: float
    reasons: list[str] = field(default_factory=list)


class AutoBalanceManager:
    """
    Manages automatic capital detection and intelligent position sizing.
    
    Features:
    - Auto-detects balance from exchange
    - Calculates free collateral after margin
    - Adapts position sizing to signal strength
    - Protects capital during drawdowns
    - Adjusts for win/loss streaks
    - Considers market volatility
    """
    
    # Capital profile tiers (USDT)
    TIERS = {
        "NANO": {"min": 0, "max": 25, "max_risk": 0.02, "min_confidence": 0.72},
        "MICRO": {"min": 25, "max": 100, "max_risk": 0.015, "min_confidence": 0.68},
        "SMALL": {"min": 100, "max": 500, "max_risk": 0.012, "min_confidence": 0.62},
        "MEDIUM": {"min": 500, "max": 2000, "max_risk": 0.01, "min_confidence": 0.58},
        "LARGE": {"min": 2000, "max": 10000, "max_risk": 0.008, "min_confidence": 0.55},
        "VERY_LARGE": {"min": 10000, "max": float('inf'), "max_risk": 0.006, "min_confidence": 0.52},
    }
    
    def __init__(
        self,
        initial_capital: float = 0.0,
        reserve_ratio: float = 0.15,  # Keep 15% as reserve
        max_daily_risk: float = 0.03,  # Max 3% daily risk
        profit_target_multiplier: float = 2.0,
    ):
        self.initial_capital = initial_capital
        self.reserve_ratio = reserve_ratio
        self.max_daily_risk = max_daily_risk
        self.profit_target_multiplier = profit_target_multiplier
        
        self.state = CapitalState()
        self._win_streak = 0
        self._loss_streak = 0
        self._daily_trades = []
        self._hourly_pnl = []
        
    def get_tier(self, balance: float) -> tuple[str, dict]:
        """Get capital tier based on balance."""
        for tier_name, tier_config in self.TIERS.items():
            if tier_config["min"] <= balance < tier_config["max"]:
                return tier_name, tier_config
        return "VERY_LARGE", self.TIERS["VERY_LARGE"]
    
    def update_from_exchange(self, balance_data: dict[str, Any]) -> None:
        """Update capital state from exchange balance data."""
        try:
            # Extract balance information
            total = float(balance_data.get("total", 0.0))
            free = float(balance_data.get("free", 0.0))
            used = float(balance_data.get("used", 0.0))
            unrealized_pnl = float(balance_data.get("unrealized_pnl", 0.0))
            
            self.state.total_balance = total
            self.state.available_balance = free
            self.state.margin_used = used
            self.state.unrealized_pnl = unrealized_pnl
            self.state.free_collateral = free - (free * self.reserve_ratio)  # Reserve for fees/spread
            
            # Update peak and drawdown
            if total > self.state.peak_balance:
                self.state.peak_balance = total
            
            if self.state.peak_balance > 0:
                self.state.current_drawdown = (self.state.peak_balance - total) / self.state.peak_balance
                self.state.max_drawdown_seen = max(self.state.max_drawdown_seen, self.state.current_drawdown)
            
            self.state.last_update = datetime.now(timezone.utc)
            
            logger.debug(
                f"AutoBalance updated: total={total:.2f} free={free:.2f} "
                f"drawdown={self.state.current_drawdown:.2%} tier={self.get_tier(total)[0]}"
            )
        except Exception as e:
            logger.error(f"Failed to update balance from exchange: {e}")
    
    def calculate_position_size(
        self,
        signal_confidence: float,
        signal_side: str,
        current_price: float,
        stop_loss_price: float,
        atr: float,
        market_volatility: float = 1.0,
        existing_positions: int = 0,
    ) -> PositionSizeRecommendation:
        """
        Calculate optimal position size based on multiple factors.
        
        Args:
            signal_confidence: Confidence score of the signal (0.0-1.0)
            signal_side: "BUY" or "SELL"
            current_price: Current market price
            stop_loss_price: Stop loss price
            atr: Average True Range
            market_volatility: Current market volatility multiplier (1.0 = normal)
            existing_positions: Number of existing open positions
        
        Returns:
            PositionSizeRecommendation with sizing details and reasons
        """
        reasons = []
        
        # Base capital
        tier_name, tier_config = self.get_tier(self.state.total_balance)
        operating_capital = self.state.free_collateral
        
        # 1. Capital tier adjustment
        max_risk = tier_config["max_risk"]
        reasons.append(f"Tier: {tier_name} (max_risk: {max_risk:.1%})")
        
        # 2. Drawdown protection
        drawdown_mult = 1.0
        if self.state.current_drawdown > 0.10:  # 10% drawdown
            drawdown_mult = 0.5  # Reduce position size by 50%
            reasons.append(f"Drawdown protection: {self.state.current_drawdown:.1%} -> 50% size")
        elif self.state.current_drawdown > 0.05:  # 5% drawdown
            drawdown_mult = 0.75  # Reduce position size by 25%
            reasons.append(f"Drawdown protection: {self.state.current_drawdown:.1%} -> 75% size")
        
        # 3. Streak adjustment
        streak_mult = 1.0
        if self._win_streak >= 3:
            # On winning streak - can be slightly more aggressive
            streak_mult = 1.1
            reasons.append(f"Win streak ({self._win_streak}): +10% size")
        elif self._loss_streak >= 2:
            # On losing streak - reduce size
            streak_mult = max(0.5, 1.0 - (self._loss_streak * 0.15))
            reasons.append(f"Loss streak ({self._loss_streak}): {streak_mult:.0%} size")
        
        # 4. Confidence-based sizing
        confidence_mult = 1.0
        if signal_confidence >= 0.80:
            confidence_mult = 1.15  # Strong signal
            reasons.append(f"High confidence ({signal_confidence:.0%}): +15% size")
        elif signal_confidence >= 0.70:
            confidence_mult = 1.05
            reasons.append(f"Good confidence ({signal_confidence:.0%}): +5% size")
        elif signal_confidence < 0.55:
            confidence_mult = 0.80  # Weak signal
            reasons.append(f"Weak confidence ({signal_confidence:.0%}): 80% size")
        
        # 5. Volatility adjustment
        volatility_mult = 1.0
        if market_volatility > 1.5:  # High volatility
            volatility_mult = 0.70
            reasons.append(f"High volatility: 70% size")
        elif market_volatility > 1.2:
            volatility_mult = 0.85
            reasons.append(f"Elevated volatility: 85% size")
        elif market_volatility < 0.8:  # Low volatility - can size up
            volatility_mult = 1.1
            reasons.append(f"Low volatility: 110% size")
        
        # 6. Daily risk limit
        daily_risk_pct = self._get_daily_risk_percentage()
        if daily_risk_pct > self.max_daily_risk * 0.8:
            # Approaching daily risk limit
            reasons.append(f"Daily risk at {daily_risk_pct:.1%}: reducing size")
            streak_mult *= 0.5
        
        # 7. Position limit check
        if existing_positions > 0:
            reasons.append(f"Existing positions: {existing_positions} -> reducing size")
            position_mult = max(0.3, 1.0 - (existing_positions * 0.25))
            streak_mult *= position_mult
        
        # Calculate final risk amount
        risk_distance = abs(current_price - stop_loss_price) / current_price
        risk_per_trade = max_risk * operating_capital
        
        # Apply all multipliers
        effective_risk = risk_per_trade * drawdown_mult * streak_mult * confidence_mult * volatility_mult
        
        # Calculate position size
        if risk_distance > 0:
            notional_value = effective_risk / risk_distance
        else:
            # Fallback: use ATR for stop distance
            notional_value = effective_risk / (atr / current_price) if atr > 0 else operating_capital * max_risk
        
        # Calculate quantity
        quantity = notional_value / current_price if current_price > 0 else 0
        
        # Apply hard limits
        min_trade = tier_config["min"] * 0.01  # Minimum 1% of tier min
        max_trade = operating_capital * 0.95  # Maximum 95% of available
        
        if notional_value < min_trade:
            reasons.append(f"Below minimum trade size ({min_trade:.2f} USDT)")
            notional_value = 0
            quantity = 0
        elif notional_value > max_trade:
            reasons.append(f"Capped at max trade size ({max_trade:.2f} USDT)")
            notional_value = max_trade
            quantity = notional_value / current_price
        
        return PositionSizeRecommendation(
            quantity=quantity,
            notional_value=notional_value,
            risk_amount=effective_risk,
            risk_percentage=effective_risk / operating_capital if operating_capital > 0 else 0,
            confidence_multiplier=confidence_mult,
            drawdown_multiplier=drawdown_mult,
            volatility_multiplier=volatility_mult,
            streak_multiplier=streak_mult,
            reasons=reasons,
        )
    
    def _get_daily_risk_percentage(self) -> float:
        """Calculate percentage of capital risked today."""
        if self.state.total_balance <= 0:
            return 0.0
        
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        daily_pnl = sum(p for t, p in self._daily_trades if t >= today_start)
        
        return abs(daily_pnl) / self.state.total_balance
    
    def record_trade_result(self, pnl: float) -> None:
        """Record trade result for streak and daily tracking."""
        self.state.pnl_today += pnl
        self.state.trades_today += 1
        self._daily_trades.append((datetime.now(timezone.utc), pnl))
        
        # Update streaks
        if pnl > 0:
            self._win_streak += 1
            self._loss_streak = 0
        elif pnl < 0:
            self._loss_streak += 1
            self._win_streak = 0
        else:
            # Break even - reset both streaks
            self._win_streak = 0
            self._loss_streak = 0
        
        # Clean old daily trades (keep last 24h)
        cutoff = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        self._daily_trades = [(t, p) for t, p in self._daily_trades if t >= cutoff]
    
    def can_trade(self, signal_confidence: float) -> tuple[bool, str]:
        """
        Check if trading is allowed based on current conditions.
        
        Returns:
            (can_trade: bool, reason: str)
        """
        tier_name, tier_config = self.get_tier(self.state.total_balance)
        
        # Check minimum confidence
        if signal_confidence < tier_config["min_confidence"]:
            return False, f"Confidence {signal_confidence:.0%} below tier minimum {tier_config['min_confidence']:.0%}"
        
        # Check drawdown
        if self.state.current_drawdown > 0.20:  # 20% drawdown
            return False, f"Drawdown protection: {self.state.current_drawdown:.1%} > 20%"
        
        # Check daily risk
        daily_risk = self._get_daily_risk_percentage()
        if daily_risk > self.max_daily_risk:
            return False, f"Daily risk limit reached: {daily_risk:.1%} > {self.max_daily_risk:.1%}"
        
        # Check loss streak
        if self._loss_streak >= 5:
            return False, f"Loss streak protection: {self._loss_streak} consecutive losses"
        
        # Check available capital
        if self.state.free_collateral < tier_config["min"] * 0.1:
            return False, f"Insufficient capital: {self.state.free_collateral:.2f} < {tier_config['min'] * 0.1:.2f}"
        
        return True, "OK"
    
    def get_recommended_confidence_threshold(self) -> float:
        """Get recommended minimum confidence based on current conditions."""
        tier_name, tier_config = self.get_tier(self.state.total_balance)
        base_threshold = tier_config["min_confidence"]
        
        # Increase threshold during drawdown
        if self.state.current_drawdown > 0.10:
            base_threshold += 0.05
        elif self.state.current_drawdown > 0.05:
            base_threshold += 0.03
        
        # Increase threshold during loss streak
        if self._loss_streak >= 3:
            base_threshold += 0.05
        elif self._loss_streak >= 2:
            base_threshold += 0.03
        
        # Decrease threshold during win streak (but not too much)
        if self._win_streak >= 5:
            base_threshold = max(tier_config["min_confidence"], base_threshold - 0.02)
        
        return min(base_threshold, 0.85)  # Cap at 85%
    
    def get_status(self) -> dict[str, Any]:
        """Get current status for dashboard/reporting."""
        tier_name, tier_config = self.get_tier(self.state.total_balance)
        
        return {
            "capital_tier": tier_name,
            "total_balance": self.state.total_balance,
            "available_balance": self.state.available_balance,
            "free_collateral": self.state.free_collateral,
            "unrealized_pnl": self.state.unrealized_pnl,
            "current_drawdown": self.state.current_drawdown,
            "max_drawdown_seen": self.state.max_drawdown_seen,
            "win_streak": self._win_streak,
            "loss_streak": self._loss_streak,
            "trades_today": self.state.trades_today,
            "pnl_today": self.state.pnl_today,
            "daily_risk_pct": self._get_daily_risk_percentage(),
            "recommended_min_confidence": self.get_recommended_confidence_threshold(),
        }