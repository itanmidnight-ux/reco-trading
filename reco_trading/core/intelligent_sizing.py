from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CapitalConfig:
    """Configuration for intelligent capital management."""
    min_capital_per_trade: float = 10.0
    max_capital_per_trade: float = 1000.0
    risk_per_trade_percent: float = 1.0
    max_daily_risk_percent: float = 5.0
    max_total_exposure_percent: float = 80.0
    min_reserve_percent: float = 20.0
    allow_compounding: bool = True
    conservative_mode: bool = False


@dataclass
class MarketCondition:
    """Current market condition assessment."""
    volatility: str = "NORMAL"
    trend: str = "NEUTRAL"
    volume: str = "NORMAL"
    liquidity: str = "NORMAL"
    sentiment: str = "NEUTRAL"
    overall_score: float = 0.5


class IntelligentPositionSizer:
    """
    Intelligent position sizing based on:
    - Available capital
    - Market conditions
    - Risk tolerance
    - Current drawdown
    """

    def __init__(self, config: CapitalConfig | None = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or CapitalConfig()
        self.initial_capital = 0.0
        self.current_capital = 0.0
        self.daily_risk_used = 0.0
        self.trades_today = 0
        self.last_reset = datetime.now(timezone.utc)
        self._win_streak = 0
        self._loss_streak = 0

    def initialize_capital(self, capital: float) -> None:
        """Initialize with starting capital."""
        self.initial_capital = capital
        self.current_capital = capital
        self.logger.info(f"Position Sizer initialized with capital: ${capital:.2f}")

    def update_capital(self, new_capital: float) -> None:
        """Update current capital after trades."""
        self.current_capital = new_capital

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_percent: float,
        market_condition: MarketCondition | None = None,
        confidence: float = 0.5,
    ) -> dict[str, Any]:
        """Calculate optimal position size based on multiple factors."""

        if self._should_reset_daily():
            self._reset_daily()

        available_capital = self._get_available_capital()
        
        if available_capital < self.config.min_capital_per_trade:
            return {
                "size": 0,
                "reason": "insufficient_capital",
                "available": available_capital,
            }

        base_size = self._calculate_base_size(available_capital)

        market_multiplier = self._get_market_multiplier(market_condition)
        
        confidence_multiplier = self._get_confidence_multiplier(confidence)
        
        risk_multiplier = self._get_risk_multiplier()
        
        final_size = (
            base_size 
            * market_multiplier 
            * confidence_multiplier 
            * risk_multiplier
        )

        final_size = max(
            self.config.min_capital_per_trade,
            min(final_size, self.config.max_capital_per_trade)
        )

        if final_size > available_capital:
            final_size = available_capital

        position_value = final_size
        quantity = position_value / entry_price if entry_price > 0 else 0

        risk_amount = position_value * (stop_loss_percent / 100)
        
        self.daily_risk_used += risk_amount
        self.trades_today += 1

        return {
            "size": final_size,
            "quantity": quantity,
            "position_value": position_value,
            "risk_amount": risk_amount,
            "risk_percent": stop_loss_percent,
            "market_multiplier": market_multiplier,
            "confidence_multiplier": confidence_multiplier,
            "risk_multiplier": risk_multiplier,
            "available_capital": available_capital,
            "reason": "normal",
        }

    def _calculate_base_size(self, available_capital: float) -> float:
        """Calculate base position size."""
        
        if self.config.conservative_mode:
            base_percent = 0.02
        else:
            base_percent = 0.05

        if self.config.allow_compounding:
            compound_factor = (self.current_capital / self.initial_capital) if self.initial_capital > 0 else 1.0
            compound_factor = max(0.5, min(compound_factor, 2.0))
            base_size = available_capital * base_percent * compound_factor
        else:
            base_size = available_capital * base_percent

        return base_size

    def _get_market_multiplier(self, condition: MarketCondition | None) -> float:
        """Adjust position size based on market conditions."""
        
        if condition is None:
            return 1.0

        multiplier = 1.0

        if condition.volatility == "LOW":
            multiplier *= 1.2
        elif condition.volatility == "HIGH":
            multiplier *= 0.7
        elif condition.volatility == "EXTREME":
            multiplier *= 0.4

        if condition.trend == "BULLISH":
            multiplier *= 1.2
        elif condition.trend == "BEARISH":
            multiplier *= 0.8

        if condition.volume == "HIGH":
            multiplier *= 1.1
        elif condition.volume == "LOW":
            multiplier *= 0.8

        if condition.liquidity == "HIGH":
            multiplier *= 1.1
        elif condition.liquidity == "LOW":
            multiplier *= 0.7

        return max(0.3, min(multiplier, 1.5))

    def _get_confidence_multiplier(self, confidence: float) -> float:
        """Adjust position size based on signal confidence."""
        
        if confidence >= 0.85:
            return 1.3
        elif confidence >= 0.70:
            return 1.1
        elif confidence >= 0.50:
            return 0.8
        else:
            return 0.5

    def _get_risk_multiplier(self) -> float:
        """Adjust position size based on current risk state."""
        
        multiplier = 1.0

        if self._loss_streak >= 3:
            multiplier *= 0.6
        elif self._loss_streak >= 2:
            multiplier *= 0.8

        if self._win_streak >= 5:
            multiplier *= 1.2
        elif self._win_streak >= 3:
            multiplier *= 1.1

        daily_risk_limit = self.config.max_daily_risk_percent
        if self.daily_risk_used >= daily_risk_limit:
            multiplier *= 0.0

        return max(0.0, min(multiplier, 1.5))

    def _get_available_capital(self) -> float:
        """Get available capital for trading."""
        
        reserve = self.current_capital * (self.config.min_reserve_percent / 100)
        max_exposure = self.current_capital * (self.config.max_total_exposure_percent / 100)
        
        available = max_exposure - reserve
        
        return max(0, min(available, self.current_capital - reserve))

    def _should_reset_daily(self) -> bool:
        """Check if daily counters should reset."""
        
        now = datetime.now(timezone.utc)
        
        if now.date() > self.last_reset.date():
            return True
        
        return False

    def _reset_daily(self) -> None:
        """Reset daily counters."""
        
        self.daily_risk_used = 0.0
        self.trades_today = 0
        self.last_reset = datetime.now(timezone.utc)
        
        self.logger.info("Daily position sizing counters reset")

    def record_trade_result(self, pnl_percent: float, won: bool) -> None:
        """Record trade result for adaptive sizing."""
        
        if won:
            self._win_streak += 1
            self._loss_streak = 0
        else:
            self._loss_streak += 1
            self._win_streak = 0

    def calculate_kelly_position_size(
        self,
        win_rate: float,
        avg_win_percent: float,
        avg_loss_percent: float,
        confidence: float = 1.0,
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Returns the fraction of capital to risk (0-1).
        """
        if win_rate <= 0 or avg_loss_percent <= 0:
            return self.config.risk_per_trade_percent / 100
        
        win_loss_ratio = avg_win_percent / avg_loss_percent
        
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        kelly = max(0.0, min(kelly, 0.25))
        
        half_kelly = kelly / 2
        
        quarter_kelly = half_kelly / 2
        
        if self.config.conservative_mode:
            optimal_fraction = quarter_kelly
        elif confidence >= 0.8:
            optimal_fraction = half_kelly
        else:
            optimal_fraction = quarter_kelly
        
        optimal_fraction = min(optimal_fraction, self.config.risk_per_trade_percent / 100)
        
        self.logger.info(
            f"Kelly Criterion: Full={kelly:.2%}, Half={half_kelly:.2%}, "
            f"Quarter={quarter_kelly:.2%}, Using={optimal_fraction:.2%} "
            f"(win_rate={win_rate:.1%}, win_loss_ratio={win_loss_ratio:.2f})"
        )
        
        return optimal_fraction

    def get_status(self) -> dict:
        """Get current position sizing status."""
        
        return {
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "profit_percent": ((self.current_capital - self.initial_capital) / self.initial_capital * 100) if self.initial_capital > 0 else 0,
            "daily_risk_used": self.daily_risk_used,
            "daily_risk_limit": self.config.max_daily_risk_percent,
            "trades_today": self.trades_today,
            "win_streak": self._win_streak,
            "loss_streak": self._loss_streak,
            "available_capital": self._get_available_capital(),
        }


class SmartMarketSelector:
    """
    Automatically selects the best market conditions and pairs to trade.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._market_history: dict[str, list[dict]] = {}
        self._preferred_pairs: list[str] = []
        self._avoid_pairs: list[str] = []

    def analyze_market_conditions(
        self,
        tickers: dict,
        ohlcv_data: dict[str, list],
    ) -> dict[str, Any]:
        """Analyze all markets and return best conditions."""

        market_analysis = []

        for symbol, ticker in tickers.items():
            if symbol in self._avoid_pairs:
                continue

            try:
                ohlcv = ohlcv_data.get(symbol, [])
                
                if not ohlcv or len(ohlcv) < 24:
                    continue

                closes = [c[4] for c in ohlcv]
                volumes = [c[5] for c in ohlcv]

                analysis = self._analyze_single_market(symbol, ticker, closes, volumes)
                market_analysis.append(analysis)

            except Exception as e:
                self.logger.warning(f"Failed to analyze {symbol}: {e}")

        market_analysis.sort(key=lambda x: x["score"], reverse=True)

        best_market = market_analysis[0] if market_analysis else None

        return {
            "best_market": best_market,
            "all_markets": market_analysis[:10],
            "recommended_action": self._get_recommended_action(best_market),
        }

    def _analyze_single_market(
        self,
        symbol: str,
        ticker: dict,
        closes: list[float],
        volumes: list[float],
    ) -> dict:
        """Analyze a single market."""

        import numpy as np

        current_price = float(ticker.get("last", 0))
        volume_24h = float(ticker.get("quoteVolume", 0))
        change_24h = float(ticker.get("percentage", 0))

        returns = np.diff(np.log(closes))
        volatility = np.std(returns) * np.sqrt(288)

        if len(volumes) >= 24:
            avg_volume = np.mean(volumes[-24:])
            volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1
        else:
            volume_ratio = 1

        if len(closes) >= 20:
            sma_20 = np.mean(closes[-20:])
            price_vs_sma = (current_price - sma_20) / sma_20 if sma_20 > 0 else 0
        else:
            price_vs_sma = 0

        score = 0

        if volume_24h > 10_000_000:
            score += 30
        elif volume_24h > 1_000_000:
            score += 20

        if abs(change_24h) < 3:
            score += 20
        elif abs(change_24h) < 5:
            score += 10

        if volatility < 0.03:
            score += 20
        elif volatility < 0.06:
            score += 10

        if volume_ratio > 1.2:
            score += 15

        if price_vs_sma > 0:
            score += 10

        if volatility > 0.10:
            condition = "HIGH_VOLATILITY"
        elif volatility > 0.05:
            condition = "NORMAL"
        else:
            condition = "LOW_VOLATILITY"

        if change_24h > 2:
            trend = "BULLISH"
        elif change_24h < -2:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"

        return {
            "symbol": symbol,
            "price": current_price,
            "volume_24h": volume_24h,
            "change_24h": change_24h,
            "volatility": volatility,
            "volume_ratio": volume_ratio,
            "trend": trend,
            "condition": condition,
            "score": score,
        }

    def _get_recommended_action(self, best_market: dict | None) -> str:
        """Get recommended trading action based on market analysis."""

        if not best_market:
            return "NO_MARKET"

        score = best_market.get("score", 0)
        volatility = best_market.get("volatility", 0)

        if score >= 80 and volatility < 0.05:
            return "TRADE_AGGRESSIVE"
        elif score >= 60:
            return "TRADE_NORMAL"
        elif score >= 40:
            return "TRADE_CONSERVATIVE"
        else:
            return "NO_TRADE"

    def set_preferred_pairs(self, pairs: list[str]) -> None:
        """Set preferred trading pairs."""
        self._preferred_pairs = pairs
        self.logger.info(f"Preferred pairs set: {pairs}")

    def add_to_avoid_list(self, pair: str, reason: str = "") -> None:
        """Add pair to avoid list."""
        self._avoid_pairs.append(pair)
        self.logger.info(f"Added to avoid list: {pair} ({reason})")


class DynamicFilterAdjuster:
    """
    Automatically adjusts trading filters based on market conditions.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._filter_history: list[dict] = []
        self._current_filters = {
            "min_confidence": 0.5,
            "min_volume": 1_000_000,
            "max_spread": 0.5,
            "min_volatility": 0.0,
            "max_volatility": 0.15,
        }

    def adjust_filters(
        self,
        market_condition: str,
        recent_performance: dict | None = None,
    ) -> dict:
        """Adjust filters based on market conditions."""

        original_filters = self._current_filters.copy()

        if market_condition == "HIGH_VOLATILITY":
            self._current_filters["min_confidence"] = min(0.7, self._current_filters["min_confidence"] + 0.1)
            self._current_filters["max_volatility"] = 0.10
            self._current_filters["min_volume"] = self._current_filters["min_volume"] * 1.5

        elif market_condition == "LOW_VOLATILITY":
            self._current_filters["min_confidence"] = max(0.4, self._current_filters["min_confidence"] - 0.05)
            self._current_filters["max_volatility"] = 0.20

        elif market_condition == "TRENDING_UP":
            self._current_filters["min_confidence"] = max(0.45, self._current_filters["min_confidence"] - 0.05)

        elif market_condition == "TRENDING_DOWN":
            self._current_filters["min_confidence"] = min(0.65, self._current_filters["min_confidence"] + 0.1)

        if recent_performance:
            if recent_performance.get("win_rate", 0) < 0.4:
                self._current_filters["min_confidence"] = min(0.8, self._current_filters["min_confidence"] + 0.1)
                self.logger.warning("Low win rate detected, increasing filter strictness")

            elif recent_performance.get("win_rate", 0) > 0.7:
                self._current_filters["min_confidence"] = max(0.4, self._current_filters["min_confidence"] - 0.05)
                self.logger.info("High win rate detected, relaxing filters slightly")

        if self._current_filters != original_filters:
            self._filter_history.append({
                "timestamp": datetime.now(timezone.utc),
                "original": original_filters,
                "new": self._current_filters,
                "reason": market_condition,
            })

        return self._current_filters.copy()

    def get_current_filters(self) -> dict:
        """Get current filter settings."""
        return self._current_filters.copy()

    def reset_filters(self) -> dict:
        """Reset filters to defaults."""
        default_filters = {
            "min_confidence": 0.5,
            "min_volume": 1_000_000,
            "max_spread": 0.5,
            "min_volatility": 0.0,
            "max_volatility": 0.15,
        }
        self._current_filters = default_filters
        return default_filters.copy()
