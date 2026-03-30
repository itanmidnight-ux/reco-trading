from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any
from enum import Enum


logger = logging.getLogger(__name__)


class EmergencyLevel(Enum):
    NORMAL = "normal"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class CircuitBreakerConfig:
    max_consecutive_losses: int = 5
    max_drawdown_percent: float = 10.0
    max_daily_loss_percent: float = 5.0
    max_exposure_percent: float = 0.8
    max_correlation: float = 0.85
    emergency_stop_triggered: bool = False
    cooldown_seconds: int = 300


@dataclass
class KillSwitchConfig:
    enabled: bool = True
    max_drawdown_percent: float = 15.0
    max_daily_loss_percent: float = 8.0
    max_consecutive_losses: int = 10
    max_session_loss_percent: float = 20.0
    auto_restart_after_seconds: int = 3600
    require_manual_reset: bool = True


class EmergencySystem:
    """
    Emergency and safety system for the trading bot.
    Includes kill-switch, circuit breakers, and emergency procedures.
    """
    
    def __init__(
        self,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
        kill_switch_config: KillSwitchConfig | None = None,
    ):
        self.logger = logging.getLogger(__name__)
        
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()
        self.kill_switch_config = kill_switch_config or KillSwitchConfig()
        
        self._emergency_level: EmergencyLevel = EmergencyLevel.NORMAL
        self._trading_paused: bool = False
        self._pause_reason: str | None = None
        self._pause_until: datetime | None = None
        
        self._consecutive_losses: int = 0
        self._session_start: datetime = datetime.now(timezone.utc)
        self._session_pnl: float = 0.0
        self._session_trades: int = 0
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._last_reset: datetime = datetime.now(timezone.utc).date()
        
        self._equity_peak: float = 0.0
        self._equity_start: float = 0.0
        
        self._order_reconciliation_errors: int = 0
        self._last_reconciliation: datetime | None = None
        
        self._kill_switch_triggered: bool = False
        self._kill_switch_time: datetime | None = None
        
        self._emergency_listeners: list[callable] = []
        
        self.logger.info("Emergency system initialized")
    
    def add_emergency_listener(self, callback: callable) -> None:
        """Add a listener for emergency events."""
        self._emergency_listeners.append(callback)
    
    def _notify_emergency_listeners(self, level: EmergencyLevel, message: str) -> None:
        """Notify all emergency listeners."""
        for listener in self._emergency_listeners:
            try:
                listener(level, message)
            except Exception as e:
                self.logger.error(f"Emergency listener error: {e}")
    
    def initialize_capital(self, capital: float) -> None:
        """Initialize capital tracking."""
        self._equity_start = capital
        self._equity_peak = capital
    
    def update_equity(self, current_equity: float) -> None:
        """Update current equity for tracking."""
        if self._equity_peak == 0:
            self._equity_peak = current_equity
        else:
            self._equity_peak = max(self._equity_peak, current_equity)
    
    def record_trade(self, pnl: float, won: bool) -> None:
        """Record a trade for monitoring."""
        self._session_pnl += pnl
        self._session_trades += 1
        self._daily_pnl += pnl
        self._daily_trades += 1
        
        if won:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
        
        self._check_emergency_conditions()
    
    def _check_emergency_conditions(self) -> None:
        """Check all emergency conditions."""
        if self._kill_switch_triggered:
            return
        
        current_level = self._emergency_level
        
        drawdown = 0.0
        if self._equity_peak > 0 and self._equity_start > 0:
            drawdown = ((self._equity_peak - self._equity_start) / self._equity_peak) * 100
        
        if (self.kill_switch_config.enabled and 
            (drawdown >= self.kill_switch_config.max_drawdown_percent or
             abs(self._daily_pnl) / max(self._equity_start, 1) * 100 >= self.kill_switch_config.max_daily_loss_percent or
             self._consecutive_losses >= self.kill_switch_config.max_consecutive_losses or
             abs(self._session_pnl) / max(self._equity_start, 1) * 100 >= self.kill_switch_config.max_session_loss_percent)):
            
            self._trigger_kill_switch(drawdown)
            return
        
        if self._consecutive_losses >= self.circuit_breaker_config.max_consecutive_losses:
            self._set_emergency_level(EmergencyLevel.CRITICAL, f"Max consecutive losses: {self._consecutive_losses}")
            self._pause_trading("circuit_breaker_losses", timedelta(seconds=self.circuit_breaker_config.cooldown_seconds))
        elif drawdown >= self.circuit_breaker_config.max_drawdown_percent * 0.8:
            self._set_emergency_level(EmergencyLevel.WARNING, f"Drawdown: {drawdown:.1f}%")
        elif self._consecutive_losses >= 3:
            self._set_emergency_level(EmergencyLevel.CAUTION, f"Consecutive losses: {self._consecutive_losses}")
        
        if current_level != self._emergency_level:
            self.logger.warning(f"Emergency level changed: {current_level.value} -> {self._emergency_level.value}")
    
    def _trigger_kill_switch(self, drawdown: float) -> None:
        """Trigger the kill switch."""
        self._kill_switch_triggered = True
        self._kill_switch_time = datetime.now(timezone.utc)
        self._trading_paused = True
        self._emergency_level = EmergencyLevel.EMERGENCY
        
        reason = f"Kill-switch triggered - Drawdown: {drawdown:.1f}%, Daily Loss: {abs(self._daily_pnl):.2f}, Consecutive Losses: {self._consecutive_losses}"
        
        self.logger.critical(f"!!! KILL-SWITCH TRIGGERED !!! {reason}")
        self._notify_emergency_listeners(EmergencyLevel.EMERGENCY, reason)
    
    def _set_emergency_level(self, level: EmergencyLevel, reason: str) -> None:
        """Set the emergency level."""
        if level != self._emergency_level:
            self._emergency_level = level
            self.logger.warning(f"Emergency level: {level.value} - {reason}")
            self._notify_emergency_listeners(level, reason)
    
    def _pause_trading(self, reason: str, duration: timedelta) -> None:
        """Pause trading for a specified duration."""
        self._trading_paused = True
        self._pause_reason = reason
        self._pause_until = datetime.now(timezone.utc) + duration
        self.logger.warning(f"Trading paused: {reason} until {self._pause_until}")
    
    def is_trading_allowed(self) -> tuple[bool, str]:
        """Check if trading is currently allowed."""
        if self._kill_switch_triggered:
            if self.kill_switch_config.require_manual_reset:
                return False, "KILL_SWITCH_TRIGGERED"
            
            if self._kill_switch_time:
                elapsed = datetime.now(timezone.utc) - self._kill_switch_time
                if elapsed.total_seconds() >= self.kill_switch_config.auto_restart_after_seconds:
                    self.logger.info("Kill-switch auto-restart after cooldown")
                    self.reset_kill_switch()
                else:
                    remaining = self.kill_switch_config.auto_restart_after_seconds - elapsed.total_seconds()
                    return False, f"KILL_SWITCH_COOLDOWN ({int(remaining)}s)"
        
        if self._trading_paused:
            if self._pause_until and datetime.now(timezone.utc) > self._pause_until:
                self._resume_trading()
            else:
                return False, f"PAUSED: {self._pause_reason}"
        
        return True, "OK"
    
    def _resume_trading(self) -> None:
        """Resume trading after pause."""
        self._trading_paused = False
        self._pause_reason = None
        self._pause_until = None
        self.logger.info("Trading resumed")
    
    def reset_kill_switch(self) -> None:
        """Reset the kill switch."""
        self._kill_switch_triggered = False
        self._kill_switch_time = None
        self._trading_paused = False
        self._emergency_level = EmergencyLevel.NORMAL
        self._consecutive_losses = 0
        self.logger.info("Kill-switch reset")
    
    def record_reconciliation_error(self) -> None:
        """Record an order reconciliation error."""
        self._order_reconciliation_errors += 1
        if self._order_reconciliation_errors >= 3:
            self._set_emergency_level(EmergencyLevel.WARNING, f"Order reconciliation errors: {self._order_reconciliation_errors}")
    
    def reset_daily_metrics(self) -> None:
        """Reset daily metrics."""
        today = datetime.now(timezone.utc).date()
        if today > self._last_reset:
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._last_reset = today
            self.logger.info("Daily metrics reset")
    
    def check_order_reconciliation_needed(self) -> bool:
        """Check if order reconciliation is needed."""
        now = datetime.now(timezone.utc)
        if self._last_reconciliation is None:
            return True
        
        elapsed = (now - self._last_reconciliation).total_seconds()
        return elapsed >= 300
    
    def record_reconciliation(self) -> None:
        """Record a successful reconciliation."""
        self._last_reconciliation = datetime.now(timezone.utc)
        self._order_reconciliation_errors = 0
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly Criterion for position sizing.
        Returns the optimal fraction of capital to risk.
        """
        if win_rate <= 0 or avg_loss <= 0:
            return 0.0
        
        win_loss_ratio = avg_win / avg_loss
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        kelly = max(0.0, min(kelly, 0.25))
        
        half_kelly = kelly / 2
        self.logger.info(f"Kelly Criterion: Full={kelly:.2%}, Half-Kelly={half_kelly:.2%}")
        
        return half_kelly
    
    def get_status(self) -> dict[str, Any]:
        """Get current emergency system status."""
        trading_allowed, reason = self.is_trading_allowed()
        
        drawdown = 0.0
        if self._equity_peak > 0 and self._equity_start > 0:
            drawdown = max(0.0, (self._equity_peak - self._equity_start) / self._equity_peak * 100)
        
        return {
            "emergency_level": self._emergency_level.value,
            "trading_allowed": trading_allowed,
            "pause_reason": reason,
            "kill_switch_triggered": self._kill_switch_triggered,
            "consecutive_losses": self._consecutive_losses,
            "session_pnl": self._session_pnl,
            "session_trades": self._session_trades,
            "daily_pnl": self._daily_pnl,
            "daily_trades": self._daily_trades,
            "drawdown_percent": drawdown,
            "equity_start": self._equity_start,
            "equity_peak": self._equity_peak,
            "order_reconciliation_errors": self._order_reconciliation_errors,
            "last_reconciliation": self._last_reconciliation.isoformat() if self._last_reconciliation else None,
        }


class DataValidator:
    """
    Validates market data for anomalies and corruption.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._price_history: dict[str, list[float]] = {}
        self._volume_history: dict[str, list[float]] = {}
        self._max_price_change_threshold = 0.20
        self._min_volume_threshold = 0.0
    
    def validate_ohlcv(self, symbol: str, ohlcv: list) -> tuple[bool, str]:
        """Validate OHLCV data for anomalies."""
        if not ohlcv or len(ohlcv) < 2:
            return False, "Insufficient data"
        
        try:
            latest = ohlcv[-1]
            if len(latest) < 6:
                return False, "Invalid OHLCV format"
            
            open_price = float(latest[1])
            high_price = float(latest[2])
            low_price = float(latest[3])
            close_price = float(latest[4])
            volume = float(latest[5])
            
            if open_price <= 0 or high_price <= 0 or low_price <= 0 or close_price <= 0:
                return False, "Invalid price: non-positive value"
            
            if high_price < low_price:
                return False, "Invalid OHLC: high < low"
            
            if high_price < open_price or high_price < close_price:
                return False, "Invalid OHLC: high < open/close"
            
            if low_price > open_price or low_price > close_price:
                return False, "Invalid OHLC: low > open/close"
            
            price_change = abs(close_price - open_price) / open_price
            if price_change > self._max_price_change_threshold:
                return False, f"Price change {price_change:.1%} exceeds threshold {self._max_price_change_threshold:.1%}"
            
            if volume < self._min_volume_threshold:
                return False, f"Volume {volume} below minimum"
            
            if symbol not in self._price_history:
                self._price_history[symbol] = []
                self._volume_history[symbol] = []
            
            self._price_history[symbol].append(close_price)
            self._volume_history[symbol].append(volume)
            
            if len(self._price_history[symbol]) > 100:
                self._price_history[symbol] = self._price_history[symbol][-100:]
            if len(self._volume_history[symbol]) > 100:
                self._volume_history[symbol] = self._volume_history[symbol][-100:]
            
            return True, "OK"
        
        except (IndexError, ValueError) as e:
            return False, f"Parse error: {e}"
    
    def detect_outliers(self, symbol: str, window: int = 20) -> list[float]:
        """Detect price outliers using standard deviation."""
        if symbol not in self._price_history or len(self._price_history[symbol]) < window:
            return []
        
        prices = self._price_history[symbol][-window:]
        mean = sum(prices) / len(prices)
        variance = sum((p - mean) ** 2 for p in prices) / len(prices)
        std = variance ** 0.5
        
        outliers = []
        for i, price in enumerate(prices[-5:]):
            if abs(price - mean) > 3 * std:
                outliers.append(price)
        
        return outliers
    
    def get_data_quality_score(self, symbol: str) -> float:
        """Get data quality score (0-1)."""
        score = 1.0
        
        if symbol not in self._price_history:
            return 0.5
        
        price_count = len(self._price_history[symbol])
        if price_count < 20:
            score *= 0.5
        elif price_count < 50:
            score *= 0.8
        
        outliers = self.detect_outliers(symbol)
        if outliers:
            score *= (1.0 - len(outliers) * 0.1)
        
        return max(0.0, min(1.0, score))
