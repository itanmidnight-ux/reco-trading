from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ProtectionType(Enum):
    COOLDOWN = "cooldown"
    LOW_PROFIT_PAIRS = "low_profit_pairs"
    MAX_DRAWDOWN = "max_drawdown"
    STOPLOSS_GUARD = "stoploss_guard"


@dataclass
class ProtectionResult:
    allowed: bool
    reason: str
    lock_until: datetime | None = None


class Protection(ABC):
    def __init__(self, config: dict[str, Any]):
        self._config = config

    @abstractmethod
    def check(self, **kwargs) -> ProtectionResult:
        pass


class CooldownPeriod(Protection):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._min_duration = timedelta(minutes=config.get("cooldown_period_duration", 15))
        self._last_trade_timestamp: datetime | None = None

    def check(self, last_trade_timestamp: datetime | None = None, **kwargs) -> ProtectionResult:
        if last_trade_timestamp is None:
            return ProtectionResult(allowed=True, reason="No recent trades")
        
        self._last_trade_timestamp = last_trade_timestamp
        time_since_trade = datetime.now(timezone.utc) - last_trade_timestamp.replace(tzinfo=timezone.utc)
        
        if time_since_trade < self._min_duration:
            remaining = self._min_duration - time_since_trade
            return ProtectionResult(
                allowed=False,
                reason=f"Cooldown active, {int(remaining.total_seconds() / 60)} minutes remaining",
                lock_until=datetime.now(timezone.utc) + remaining
            )
        
        return ProtectionResult(allowed=True, reason="Cooldown complete")


class LowProfitPairs(Protection):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._min_profit = config.get("min_profit", 0.0)
        self._lookback_period = timedelta(hours=config.get("lookback_period", 24))
        self._locked_pairs: dict[str, datetime] = {}

    def check(self, pair: str, recent_trades: list[dict] | None = None, **kwargs) -> ProtectionResult:
        if recent_trades is None or not recent_trades:
            return ProtectionResult(allowed=True, reason="No recent trades")
        
        if pair in self._locked_pairs:
            if datetime.now(timezone.utc) < self._locked_pairs[pair]:
                return ProtectionResult(
                    allowed=False,
                    reason=f"Pair {pair} locked due to low profit"
                )
            else:
                del self._locked_pairs[pair]
        
        pair_trades = [t for t in recent_trades if t.get("symbol") == pair]
        if len(pair_trades) >= 3:
            profits = [t.get("pnl", 0) for t in pair_trades]
            avg_profit = sum(profits) / len(profits)
            
            if avg_profit < self._min_profit:
                lock_time = datetime.now(timezone.utc) + self._lookback_period
                self._locked_pairs[pair] = lock_time
                return ProtectionResult(
                    allowed=False,
                    reason=f"Low profit detected for {pair}, locking for {self._lookback_period}",
                    lock_until=lock_time
                )
        
        return ProtectionResult(allowed=True, reason="Profit acceptable")


class MaxDrawdownProtection(Protection):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._max_drawdown = config.get("max_drawdown", 0.15)
        self._lookback_period = timedelta(hours=config.get("lookback_period", 24))

    def check(self, current_equity: float, peak_equity: float, **kwargs) -> ProtectionResult:
        if peak_equity <= 0:
            return ProtectionResult(allowed=True, reason="No peak equity")
        
        drawdown = (peak_equity - current_equity) / peak_equity
        
        if drawdown >= self._max_drawdown:
            return ProtectionResult(
                allowed=False,
                reason=f"Max drawdown reached: {drawdown:.2%}"
            )
        
        return ProtectionResult(allowed=True, reason="Drawdown acceptable")


class StoplossGuard(Protection):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._max_stoploss = config.get("max_stoploss_count", 3)
        self._lookback_period = timedelta(minutes=config.get("lookback_period", 60))
        self._stoploss_timestamps: list[datetime] = []

    def check(self, last_trade_result: str | None = None, **kwargs) -> ProtectionResult:
        now = datetime.now(timezone.utc)
        
        self._stoploss_timestamps = [
            ts for ts in self._stoploss_timestamps
            if now - ts < self._lookback_period
        ]
        
        if last_trade_result == "STOP_LOSS_HIT":
            self._stoploss_timestamps.append(now)
        
        stoploss_count = len(self._stoploss_timestamps)
        
        if stoploss_count >= self._max_stoploss:
            return ProtectionResult(
                allowed=False,
                reason=f"Too many stoplosses: {stoploss_count} in {self._lookback_period}",
                lock_until=now + self._lookback_period
            )
        
        return ProtectionResult(allowed=True, reason="Stoploss count acceptable")


class ProtectionManager:
    def __init__(self, config: list[dict[str, Any]]):
        self._protections: list[Protection] = []
        self._load_protections(config)

    def _load_protections(self, config: list[dict[str, Any]]) -> None:
        protection_map = {
            "CooldownPeriod": CooldownPeriod,
            "LowProfitPairs": LowProfitPairs,
            "MaxDrawdownProtection": MaxDrawdownProtection,
            "StoplossGuard": StoplossGuard,
        }
        
        for prot_config in config:
            prot_type = prot_config.get("name") or prot_config.get("type")
            if prot_type in protection_map:
                self._protections.append(protection_map[prot_type](prot_config))
                logger.info(f"Loaded protection: {prot_type}")

    def check_all(self, **kwargs) -> ProtectionResult:
        for protection in self._protections:
            result = protection.check(**kwargs)
            if not result.allowed:
                return result
        
        return ProtectionResult(allowed=True, reason="All protections passed")

    def can_trade(self, **kwargs) -> ProtectionResult:
        return self.check_all(**kwargs)
