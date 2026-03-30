"""
Risk Adaptation Engine Module.
Dynamically adjusts risk parameters based on market conditions.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from reco_trading.advanced_auto_improver.market_regime_detector import MarketRegime

logger = logging.getLogger(__name__)


@dataclass
class RiskParameters:
    """Current risk parameters."""
    position_size: float
    stop_loss: float
    take_profit: float
    max_drawdown_limit: float
    leverage: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RiskAdaptationEngine:
    """Dynamically adjusts risk parameters."""

    def __init__(
        self,
        base_position_size: float = 0.1,
        base_stop_loss: float = 0.03,
        base_take_profit: float = 0.06,
        max_drawdown_limit: float = 0.2,
        base_leverage: float = 1.0,
    ):
        self.base_position_size = base_position_size
        self.base_stop_loss = base_stop_loss
        self.base_take_profit = base_take_profit
        self.max_drawdown_limit = max_drawdown_limit
        self.base_leverage = base_leverage
        
        self._current_drawdown = 0.0
        self._peak_balance = 0.0
        self._current_balance = 0.0
        self._consecutive_losses = 0
        self._recent_returns: list[float] = []
        
        self._risk_history: list[RiskParameters] = []

    def update_market_data(
        self,
        current_balance: float,
        current_drawdown: float,
        recent_pnl: float,
    ) -> None:
        """Update market data for risk calculation."""
        self._current_balance = current_balance
        
        if current_balance > self._peak_balance:
            self._peak_balance = current_balance
        
        self._current_drawdown = current_drawdown
        
        self._recent_returns.append(recent_pnl)
        if len(self._recent_returns) > 20:
            self._recent_returns.pop(0)
        
        if recent_pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    def calculate_adapted_parameters(
        self,
        regime: Optional[MarketRegime] = None,
        volatility: float = 2.0,
    ) -> RiskParameters:
        """Calculate adapted risk parameters."""
        position_size = self.base_position_size
        stop_loss = self.base_stop_loss
        take_profit = self.base_take_profit
        leverage = self.base_leverage
        
        position_size = self._adjust_for_drawdown(position_size)
        position_size = self._adjust_for_consecutive_losses(position_size)
        position_size = self._adjust_for_volatility(position_size, volatility)
        
        if regime:
            position_size = self._adjust_for_regime(position_size, regime)
            stop_loss = self._adjust_stop_loss_for_regime(stop_loss, regime)
            take_profit = self._adjust_take_profit_for_regime(take_profit, regime)
        
        position_size = max(0.01, min(position_size, 0.5))
        
        params = RiskParameters(
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_drawdown_limit=self.max_drawdown_limit,
            leverage=leverage,
        )
        
        self._risk_history.append(params)
        
        if len(self._risk_history) > 1000:
            self._risk_history.pop(0)
        
        logger.debug(f"Risk params: size={position_size:.2%}, sl={stop_loss:.2%}, tp={take_profit:.2%}")
        
        return params

    def _adjust_for_drawdown(self, position_size: float) -> float:
        """Adjust position size based on drawdown."""
        if self._current_drawdown > self.max_drawdown_limit * 0.8:
            return position_size * 0.3
        elif self._current_drawdown > self.max_drawdown_limit * 0.5:
            return position_size * 0.5
        elif self._current_drawdown > self.max_drawdown_limit * 0.3:
            return position_size * 0.7
        
        return position_size

    def _adjust_for_consecutive_losses(self, position_size: float) -> float:
        """Reduce position size after consecutive losses."""
        if self._consecutive_losses >= 5:
            return position_size * 0.2
        elif self._consecutive_losses >= 3:
            return position_size * 0.5
        elif self._consecutive_losses >= 2:
            return position_size * 0.75
        
        return position_size

    def _adjust_for_volatility(self, position_size: float, volatility: float) -> float:
        """Adjust position size based on volatility."""
        if volatility > 5:
            return position_size * 0.5
        elif volatility > 3:
            return position_size * 0.7
        elif volatility > 1:
            return position_size * 0.9
        
        return position_size

    def _adjust_for_regime(self, position_size: float, regime: MarketRegime) -> float:
        """Adjust position size based on market regime."""
        if regime == MarketRegime.HIGH_VOLATILITY:
            return position_size * 0.5
        elif regime == MarketRegime.BEAR_TREND:
            return position_size * 0.7
        elif regime == MarketRegime.LOW_VOLATILITY:
            return position_size * 1.2
        
        return position_size

    def _adjust_stop_loss_for_regime(self, stop_loss: float, regime: MarketRegime) -> float:
        """Adjust stop loss based on market regime."""
        if regime == MarketRegime.HIGH_VOLATILITY:
            return stop_loss * 1.5
        elif regime == MarketRegime.BEAR_TREND:
            return stop_loss * 1.2
        elif regime == MarketRegime.LOW_VOLATILITY:
            return stop_loss * 0.8
        
        return stop_loss

    def _adjust_take_profit_for_regime(self, take_profit: float, regime: MarketRegime) -> float:
        """Adjust take profit based on market regime."""
        if regime == MarketRegime.HIGH_VOLATILITY:
            return take_profit * 1.5
        elif regime == MarketRegime.BEAR_TREND:
            return take_profit * 0.8
        elif regime == MarketRegime.BULL_TREND:
            return take_profit * 1.2
        
        return take_profit

    def get_current_parameters(self) -> Optional[RiskParameters]:
        """Get current risk parameters."""
        if not self._risk_history:
            return None
        return self._risk_history[-1]

    def get_risk_level(self) -> str:
        """Get current risk level."""
        params = self.get_current_parameters()
        
        if not params:
            return "unknown"
        
        if params.position_size < 0.05:
            return "very_low"
        elif params.position_size < 0.1:
            return "low"
        elif params.position_size < 0.2:
            return "medium"
        else:
            return "high"

    def should_pause_trading(self) -> tuple[bool, str]:
        """Determine if trading should be paused."""
        if self._current_drawdown > self.max_drawdown_limit:
            return True, f"Max drawdown exceeded: {self._current_drawdown:.1f}%"
        
        if self._consecutive_losses >= 10:
            return True, f"Too many consecutive losses: {self._consecutive_losses}"
        
        if len(self._recent_returns) >= 10:
            recent_avg = sum(self._recent_returns) / len(self._recent_returns)
            if recent_avg < -5:
                return True, f"Poor recent performance: {recent_avg:.1f}%"
        
        return False, "Trading allowed"

    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary."""
        params = self.get_current_parameters()
        
        return {
            "current_position_size": params.position_size if params else 0,
            "current_stop_loss": params.stop_loss if params else 0,
            "current_take_profit": params.take_profit if params else 0,
            "current_drawdown": self._current_drawdown,
            "consecutive_losses": self._consecutive_losses,
            "risk_level": self.get_risk_level(),
            "should_pause": self.should_pause_trading()[0],
        }
