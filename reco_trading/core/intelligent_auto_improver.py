from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

from reco_trading.auto_improver.auto_improver import AutoImprover, AutoImproverConfig
from reco_trading.advanced_auto_improver.advanced_auto_improver import AdvancedAutoImprover, AdvancedImproverConfig

logger = logging.getLogger(__name__)


@dataclass
class TradePerformance:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0
    avg_trade_duration_minutes: float = 0.0


@dataclass
class ImprovementMetrics:
    last_optimization: datetime | None = None
    optimization_count: int = 0
    current_strategy_version: str = "1.0"
    best_win_rate: float = 0.0
    best_profit_factor: float = 0.0
    improvements_applied: list[str] = field(default_factory=list)
    rollback_count: int = 0
    failed_optimizations: int = 0


class IntelligentAutoImprover:
    """
    Integrates auto-improvement into the main bot with trade loss prevention.
    """

    def __init__(self, enabled: bool = True):
        self.logger = logging.getLogger(__name__)
        self.enabled = enabled
        
        self._basic_auto_improver: AutoImprover | None = None
        self._advanced_auto_improver: AdvancedAutoImprover | None = None
        
        self._performance = TradePerformance()
        self._metrics = ImprovementMetrics()
        
        self._optimization_task: asyncio.Task | None = None
        self._optimization_interval_hours = 6
        self._min_trades_before_optimization = 5
        
        self._loss_prevention_active = False
        self._consecutive_loss_threshold = 3
        self._emergency_optimization_triggered = False
        
        self._recent_trades: list[dict[str, Any]] = []
        self._max_recent_trades = 100
        
        if self.enabled:
            self._initialize_improvers()

    def _initialize_improvers(self) -> None:
        try:
            basic_config = AutoImproverConfig()
            basic_config.auto_deploy = False
            self._basic_auto_improver = AutoImprover(config=basic_config)
            
            advanced_config = AdvancedImproverConfig()
            advanced_config.auto_deploy = False
            self._advanced_auto_improver = AdvancedAutoImprover(config=advanced_config)
            
            self.logger.info("Auto-improvement systems initialized")
        except Exception as exc:
            self.logger.error(f"Failed to initialize auto-improvement: {exc}")
            self.enabled = False

    async def start(self) -> None:
        if not self.enabled:
            self.logger.info("Auto-improvement disabled")
            return
            
        self.logger.info("Starting auto-improvement scheduler")
        self._optimization_task = asyncio.create_task(self._optimization_loop())

    async def stop(self) -> None:
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Auto-improvement scheduler stopped")

    async def _optimization_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(3600 * self._optimization_interval_hours)
                
                if self._should_optimize():
                    await self._run_optimization()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.logger.error(f"Optimization loop error: {exc}")
                await asyncio.sleep(300)

    def _should_optimize(self) -> bool:
        if self._performance.total_trades < self._min_trades_before_optimization:
            return False
            
        if self._performance.consecutive_losses >= self._consecutive_loss_threshold:
            return True
            
        if self._metrics.last_optimization is None:
            return True
            
        time_since_optimization = datetime.now(timezone.utc) - self._metrics.last_optimization
        if time_since_optimization > timedelta(hours=self._optimization_interval_hours):
            return True
            
        return False

    async def _run_optimization(self) -> None:
        self.logger.info("Starting auto-optimization cycle")
        
        try:
            if self._advanced_auto_improver:
                result = await self._run_advanced_optimization()
                
                if result.get("success"):
                    self._metrics.last_optimization = datetime.now(timezone.utc)
                    self._metrics.optimization_count += 1
                    self._metrics.improvements_applied.append(
                        f"Optimization #{self._metrics.optimization_count} at {self._metrics.last_optimization.isoformat()}"
                    )
                    self.logger.info(f"Optimization successful: {result.get('details', {})}")
                else:
                    self._metrics.failed_optimizations += 1
                    self.logger.warning(f"Optimization failed: {result.get('error')}")
                    
        except Exception as exc:
            self._metrics.failed_optimizations += 1
            self.logger.error(f"Optimization error: {exc}")

    async def _run_advanced_optimization(self) -> dict[str, Any]:
        if not self._advanced_auto_improver:
            return {"success": False, "error": "Advanced improver not initialized"}
            
        try:
            recent_trades = self._recent_trades[-self._min_trades_before_optimization:]
            
            if len(recent_trades) < self._min_trades_before_optimization:
                return {"success": False, "error": "Not enough trades for optimization"}
            
            trades_df = self._create_trades_dataframe(recent_trades)
            
            result = await asyncio.to_thread(
                self._advanced_auto_improver.analyze_and_optimize,
                trades_df
            )
            
            return {"success": True, "details": result}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _create_trades_dataframe(self, trades: list[dict[str, Any]]) -> Any:
        try:
            import pandas as pd
            
            data = []
            for trade in trades:
                data.append({
                    "timestamp": trade.get("timestamp", datetime.now(timezone.utc)),
                    "side": trade.get("side", "BUY"),
                    "entry_price": trade.get("entry", 0),
                    "exit_price": trade.get("exit", 0),
                    "quantity": trade.get("size", 0),
                    "pnl": trade.get("pnl", 0),
                    "duration_minutes": trade.get("duration_minutes", 0),
                    "exit_reason": trade.get("exit_reason", "UNKNOWN"),
                })
            
            return pd.DataFrame(data)
        except ImportError:
            return None

    def record_trade(self, trade_data: dict[str, Any]) -> None:
        self._recent_trades.append(trade_data)
        
        if len(self._recent_trades) > self._max_recent_trades:
            self._recent_trades = self._recent_trades[-self._max_recent_trades:]
        
        self._update_performance(trade_data)
        
        if trade_data.get("pnl", 0) < 0:
            self._performance.consecutive_losses += 1
            self._performance.max_consecutive_losses = max(
                self._performance.max_consecutive_losses,
                self._performance.consecutive_losses
            )
            
            if self._performance.consecutive_losses >= self._consecutive_loss_threshold:
                self._trigger_emergency_optimization()
        else:
            self._performance.consecutive_losses = 0

    def _update_performance(self, trade_data: dict[str, Any]) -> None:
        self._performance.total_trades += 1
        
        pnl = trade_data.get("pnl", 0)
        self._performance.total_pnl += pnl
        
        if pnl > 0:
            self._performance.winning_trades += 1
            self._performance.average_win = (
                (self._performance.average_win * (self._performance.winning_trades - 1) + pnl)
                / self._performance.winning_trades
            )
            self._performance.largest_win = max(self._performance.largest_win, pnl)
        elif pnl < 0:
            self._performance.losing_trades += 1
            self._performance.average_loss = (
                (self._performance.average_loss * (self._performance.losing_trades - 1) + abs(pnl))
                / self._performance.losing_trades
            )
            self._performance.largest_loss = max(self._performance.largest_loss, abs(pnl))
        
        if self._performance.total_trades > 0:
            self._performance.win_rate = (
                self._performance.winning_trades / self._performance.total_trades * 100
            )
        
        if self._performance.average_loss > 0:
            self._performance.profit_factor = (
                self._performance.average_win / self._performance.average_loss
                if self._performance.average_loss > 0 else 0
            )

    def _trigger_emergency_optimization(self) -> None:
        if self._emergency_optimization_triggered:
            return
            
        self._emergency_optimization_triggered = True
        self.logger.warning(
            f"Emergency optimization triggered after {self._performance.consecutive_losses} consecutive losses"
        )
        
        asyncio.create_task(self._run_optimization())
        
        asyncio.get_event_loop().call_later(
            3600,
            lambda: setattr(self, "_emergency_optimization_triggered", False)
        )

    def get_performance_summary(self) -> dict[str, Any]:
        return {
            "total_trades": self._performance.total_trades,
            "winning_trades": self._performance.winning_trades,
            "losing_trades": self._performance.losing_trades,
            "win_rate": f"{self._performance.win_rate:.1f}%",
            "total_pnl": f"{self._performance.total_pnl:.2f} USDT",
            "average_win": f"{self._performance.average_win:.2f} USDT",
            "average_loss": f"{self._performance.average_loss:.2f} USDT",
            "profit_factor": f"{self._performance.profit_factor:.2f}",
            "consecutive_losses": self._performance.consecutive_losses,
            "max_consecutive_losses": self._performance.max_consecutive_losses,
        }

    def get_improvement_metrics(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "last_optimization": self._metrics.last_optimization.isoformat() if self._metrics.last_optimization else None,
            "optimization_count": self._metrics.optimization_count,
            "current_strategy_version": self._metrics.current_strategy_version,
            "best_win_rate": f"{self._metrics.best_win_rate:.1f}%",
            "best_profit_factor": f"{self._metrics.best_profit_factor:.2f}",
            "improvements_applied": self._metrics.improvements_applied[-5:],
            "rollback_count": self._metrics.rollback_count,
            "failed_optimizations": self._metrics.failed_optimizations,
        }

    def should_block_trading(self) -> bool:
        if not self.enabled:
            return False
            
        if self._performance.max_consecutive_losses >= 10:
            return True
            
        if self._performance.win_rate < 20 and self._performance.total_trades >= 20:
            return True
            
        return False
