from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AutonomousConfig:
    """Configuration for autonomous trading."""
    enabled: bool = True
    auto_position_sizing: bool = True
    auto_market_selection: bool = True
    auto_filter_adjustment: bool = True
    auto_risk_adjustment: bool = True
    decision_interval_seconds: int = 60
    min_capital: float = 10.0
    max_capital: float = 1000.0


class AutonomousTradingBrain:
    """
    Complete autonomous trading system that:
    - Self-optimizes based on performance
    - Automatically selects best market conditions
    - Adjusts position sizing intelligently
    - Adapts filters dynamically
    - Makes independent trading decisions
    """

    def __init__(self, config: AutonomousConfig | None = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or AutonomousConfig()
        
        from reco_trading.core.intelligent_sizing import (
            IntelligentPositionSizer,
            SmartMarketSelector,
            DynamicFilterAdjuster,
        )
        
        self.position_sizer = IntelligentPositionSizer()
        self.market_selector = SmartMarketSelector()
        self.filter_adjuster = DynamicFilterAdjuster()
        
        self._decision_loop_task: asyncio.Task | None = None
        self._is_running = False
        self._bot_engine = None
        
        self._performance_history: list[dict] = []
        self._last_optimization: datetime | None = None
        self._consecutive_losses = 0
        self._consecutive_wins = 0
        
        self._current_best_pair: str = "BTC/USDT"
        self._current_market_condition: str = "NORMAL"
        
        self.logger.info("Autonomous Trading Brain initialized")
    
    def set_bot_engine(self, bot_engine) -> None:
        """Set reference to bot engine for accessing market data and ML."""
        self._bot_engine = bot_engine
        self.logger.info("Autonomous Brain linked to Bot Engine")

    async def start(self) -> None:
        """Start the autonomous trading system."""
        
        if not self.config.enabled:
            self.logger.info("Autonomous trading is disabled")
            return
        
        self._is_running = True
        self._decision_loop_task = asyncio.create_task(self._decision_loop())
        self.logger.info("Autonomous Trading Brain started")

    async def stop(self) -> None:
        """Stop the autonomous trading system."""
        
        self._is_running = False
        
        if self._decision_loop_task:
            self._decision_loop_task.cancel()
            try:
                await self._decision_loop_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Autonomous Trading Brain stopped")

    async def _decision_loop(self) -> None:
        """Main decision loop that runs continuously."""
        
        while self._is_running:
            try:
                await asyncio.sleep(self.config.decision_interval_seconds)
                
                await self._analyze_performance()
                
                await self._adjust_confidence_threshold()
                
                await self._adjust_to_market()
                
                await self._optimize_parameters()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Decision loop error: {e}")
                await asyncio.sleep(10)

    async def _analyze_performance(self) -> None:
        """Analyze recent performance and adjust strategy."""
        
        try:
            recent_pnls = self._performance_history[-20:] if self._performance_history else []
            
            if not recent_pnls:
                return
            
            wins = sum(1 for p in recent_pnls if p.get("pnl", 0) > 0)
            win_rate = wins / len(recent_pnls) if recent_pnls else 0
            
            avg_win = sum(p.get("pnl", 0) for p in recent_pnls if p.get("pnl", 0) > 0) / max(wins, 1)
            avg_loss = sum(abs(p.get("pnl", 0)) for p in recent_pnls if p.get("pnl", 0) < 0) / max(len(recent_pnls) - wins, 1)
            
            self.logger.info(
                f"Performance: Win Rate: {win_rate:.1%}, "
                f"Avg Win: ${avg_win:.2f}, Avg Loss: ${avg_loss:.2f}"
            )
            
            if win_rate < 0.3:
                self._consecutive_losses += 1
                self._consecutive_wins = 0
                await self._apply_defensive_measures()
            elif win_rate > 0.7:
                self._consecutive_wins += 1
                self._consecutive_losses = 0
                await self._apply_aggressive_measures()
            else:
                self._consecutive_losses = 0
                self._consecutive_wins = 0
                
        except Exception as e:
            self.logger.error(f"Performance analysis error: {e}")

    async def _apply_defensive_measures(self) -> None:
        """Apply defensive measures after consecutive losses."""
        
        self.logger.warning("Applying defensive measures due to consecutive losses")
        
        self.filter_adjuster.adjust_filters("DEFENSIVE")
        
        if hasattr(self, 'position_sizer'):
            self.position_sizer._loss_streak = 0
            self.position_sizer._win_streak = 0

    async def _apply_aggressive_measures(self) -> None:
        """Apply aggressive measures after consecutive wins."""
        
        self.logger.info("Applying aggressive measures due to consecutive wins")

    async def _adjust_to_market(self) -> None:
        """Adjust trading parameters to current market conditions."""
        
        try:
            market_condition = "NORMAL"
            
            if hasattr(self, '_bot_engine') and self._bot_engine:
                try:
                    ml = getattr(self._bot_engine, 'enhanced_ml', None)
                    if ml and hasattr(ml, 'get_market_condition'):
                        symbol = getattr(self._bot_engine, 'symbol', 'BTC/USDT')
                        frame = getattr(self._bot_engine, '_cached_frame5', None)
                        if frame is not None and len(frame) >= 20:
                            prices = frame['close'].tolist()
                            volumes = frame['volume'].tolist() if 'volume' in frame.columns else [1.0] * len(prices)
                            mc = ml.get_market_condition(symbol, prices, volumes)
                            market_condition = mc.regime if mc else "NORMAL"
                except Exception:
                    pass
            
            self._current_market_condition = market_condition
            
            self.filter_adjuster.adjust_filters(self._current_market_condition)
            
        except Exception as e:
            self.logger.error(f"Market adjustment error: {e}")

    async def _optimize_parameters(self) -> None:
        """Optimize trading parameters periodically."""
        
        now = datetime.now(timezone.utc)
        
        if self._last_optimization is None:
            self._last_optimization = now
            return
        
        hours_since = (now - self._last_optimization).total_seconds() / 3600
        
        if hours_since >= 6:
            self.logger.info("Running parameter optimization")
            self._last_optimization = now

    def initialize_capital(self, capital: float) -> None:
        """Initialize capital for position sizing."""
        self.position_sizer.initialize_capital(capital)

    def update_capital(self, capital: float) -> None:
        """Update current capital."""
        self.position_sizer.update_capital(capital)

    def calculate_trade_size(
        self,
        entry_price: float,
        stop_loss_percent: float,
        market_condition: Any = None,
        confidence: float = 0.5,
    ) -> dict:
        """Calculate optimal trade size."""
        return self.position_sizer.calculate_position_size(
            entry_price=entry_price,
            stop_loss_percent=stop_loss_percent,
            market_condition=market_condition,
            confidence=confidence,
        )

    def select_best_market(
        self,
        tickers: dict,
        ohlcv_data: dict,
    ) -> dict:
        """Select best market to trade."""
        return self.market_selector.analyze_market_conditions(tickers, ohlcv_data)

    def record_trade(self, trade_data: dict) -> None:
        """Record trade for analysis."""
        
        pnl = trade_data.get("pnl", 0)
        
        self._performance_history.append(trade_data)
        
        if len(self._performance_history) > 1000:
            self._performance_history = self._performance_history[-500:]
        
        won = pnl > 0
        self.position_sizer.record_trade_result(
            pnl_percent=trade_data.get("pnl_percent", 0),
            won=won,
        )

    def get_current_filters(self) -> dict:
        """Get current trading filters."""
        return self.filter_adjuster.get_current_filters()

    def get_status(self) -> dict:
        """Get autonomous system status."""
        
        recent_performance = None
        if self._performance_history:
            recent = self._performance_history[-20:]
            wins = sum(1 for p in recent if p.get("pnl", 0) > 0)
            recent_performance = {
                "win_rate": wins / len(recent) if recent else 0,
                "total_trades": len(recent),
            }
        
        return {
            "enabled": self.config.enabled,
            "is_running": self._is_running,
            "current_best_pair": self._current_best_pair,
            "current_market_condition": self._current_market_condition,
            "consecutive_losses": self._consecutive_losses,
            "consecutive_wins": self._consecutive_wins,
            "last_optimization": self._last_optimization.isoformat() if self._last_optimization else None,
            "position_sizing": self.position_sizer.get_status(),
            "recent_performance": recent_performance,
            "current_filters": self.get_current_filters(),
        }

    async def _adjust_confidence_threshold(self) -> None:
        """Analyze market and adjust confidence threshold automatically."""
        
        if not hasattr(self, '_bot_engine') or not self._bot_engine:
            return
        
        try:
            settings = getattr(self._bot_engine, 'settings', None)
            if not settings or not getattr(settings, 'auto_confidence_adjustment', True):
                return
            
            current_confidence = getattr(settings, 'min_signal_confidence', 0.70)
            
            market_metrics = self._bot_engine.snapshot.get("market_regime", "UNKNOWN")
            pair = getattr(self._bot_engine, 'symbol', "BTC/USDT")
            
            pair_metrics = None
            if hasattr(self._bot_engine, 'multi_pair_manager'):
                pair_metrics = self._bot_engine.multi_pair_manager.get_pair_metrics(pair)
            
            opportunity_score = pair_metrics.opportunity_score if pair_metrics else 0.5
            volatility = pair_metrics.volatility if pair_metrics else 0.03
            
            new_confidence = current_confidence
            
            if self._consecutive_losses >= 3:
                new_confidence = min(0.85, current_confidence + 0.05)
                self.logger.warning(f"Consecutive losses detected - Increasing confidence to {new_confidence:.0%}")
            elif self._consecutive_losses >= 5:
                new_confidence = min(0.90, current_confidence + 0.10)
                self.logger.warning(f"Multiple consecutive losses - Increasing confidence to {new_confidence:.0%}")
            elif self._consecutive_wins >= 5:
                new_confidence = max(0.60, current_confidence - 0.05)
                self.logger.info(f"Strong win streak - Reducing confidence to {new_confidence:.0%} to take more trades")
            
            if market_metrics == "HIGH_VOLATILITY":
                new_confidence = max(new_confidence, 0.75)
                self.logger.info(f"High volatility market - Setting confidence to {new_confidence:.0%}")
            elif market_metrics == "LOW_VOLATILITY":
                new_confidence = min(new_confidence, 0.65)
                self.logger.info(f"Low volatility market - Setting confidence to {new_confidence:.0%}")
            
            if opportunity_score > 0.7:
                new_confidence = max(0.65, new_confidence - 0.05)
                self.logger.info(f"High opportunity score ({opportunity_score:.2f}) - Reducing confidence to {new_confidence:.0%}")
            elif opportunity_score < 0.4:
                new_confidence = min(0.80, new_confidence + 0.05)
                self.logger.info(f"Low opportunity score ({opportunity_score:.2f}) - Increasing confidence to {new_confidence:.0%}")
            
            if volatility > 0.10:
                new_confidence = min(0.85, new_confidence + 0.05)
                self.logger.info(f"High volatility ({volatility:.2%}) - Increasing confidence to {new_confidence:.0%}")
            elif volatility < 0.02:
                new_confidence = max(0.60, new_confidence - 0.05)
                self.logger.info(f"Low volatility ({volatility:.2%}) - Reducing confidence to {new_confidence:.0%}")
            
            if new_confidence != current_confidence:
                self._bot_engine.runtime_confidence_threshold = new_confidence
                self.logger.info(f">>> AUTO-ADJUSTED confidence: {current_confidence:.0%} -> {new_confidence:.0%} for {pair}")
                
                self._bot_engine.snapshot["auto_confidence_adjusted"] = True
                self._bot_engine.snapshot["previous_confidence"] = current_confidence
                self._bot_engine.snapshot["new_confidence"] = new_confidence
        
        except Exception as e:
            self.logger.error(f"Error adjusting confidence threshold: {e}")


def create_autonomous_brain(
    enabled: bool = True,
    capital: float = 1000.0,
) -> AutonomousTradingBrain:
    """Factory function to create autonomous trading brain."""
    
    config = AutonomousConfig(
        enabled=enabled,
        auto_position_sizing=True,
        auto_market_selection=True,
        auto_filter_adjustment=True,
    )
    
    brain = AutonomousTradingBrain(config)
    brain.initialize_capital(capital)
    
    return brain
