from __future__ import annotations

import asyncio
import logging
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any
from pathlib import Path

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


@dataclass
class MLRunningStats:
    """Stats for ML-based self-tuning."""
    recent_win_rates: list[float] = field(default_factory=list)
    recent_profit_factors: list[float] = field(default_factory=list)
    trend_direction: str = "stable"
    predicted_performance: float = 0.0
    confidence: float = 0.0
    optimal_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftDetectionState:
    """State for strategy drift detection."""
    baseline_win_rate: float = 0.0
    baseline_profit_factor: float = 0.0
    drift_detected: bool = False
    drift_magnitude: float = 0.0
    drift_start_time: datetime | None = None
    adaptation_applied: bool = False


class MLBasedSelfTuner:
    """Machine Learning based self-tuning for parameters."""

    def __init__(self, history_window: int = 50):
        self.history_window = history_window
        self._win_rate_history: list[float] = []
        self._profit_factor_history: list[float] = []
        self._params_history: list[dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)

    def record_performance(self, win_rate: float, profit_factor: float, params: dict[str, Any]) -> None:
        self._win_rate_history.append(win_rate)
        self._profit_factor_history.append(profit_factor)
        self._params_history.append(params.copy())

        if len(self._win_rate_history) > self.history_window:
            self._win_rate_history = self._win_rate_history[-self.history_window:]
            self._profit_factor_history = self._profit_factor_history[-self.history_window:]
            self._params_history = self._params_history[-self.history_window:]

    def predict_optimal_params(self) -> dict[str, Any]:
        """Predict optimal parameters based on historical performance."""
        if len(self._win_rate_history) < 10:
            return {"action": "wait", "reason": "insufficient_data"}

        recent_wr = self._win_rate_history[-10:]
        recent_pf = self._profit_factor_history[-10:]

        avg_wr = sum(recent_wr) / len(recent_wr)
        avg_pf = sum(recent_pf) / len(recent_pf)

        wr_trend = self._calculate_trend(recent_wr)
        pf_trend = self._calculate_trend(recent_pf)

        recommendations = {}

        if wr_trend < -0.5 and avg_wr < 40:
            recommendations["action"] = "reduce_risk"
            recommendations["confidence"] = min(abs(wr_trend), 1.0)
        elif wr_trend > 0.5 and avg_wr > 60:
            recommendations["action"] = "increase_aggression"
            recommendations["confidence"] = min(wr_trend, 1.0)
        elif pf_trend < -0.3:
            recommendations["action"] = "adjust_strategy"
            recommendations["confidence"] = min(abs(pf_trend), 1.0)
        else:
            recommendations["action"] = "maintain"
            recommendations["confidence"] = 0.8

        recommendations["predicted_win_rate"] = avg_wr + wr_trend
        recommendations["predicted_profit_factor"] = avg_pf + pf_trend

        return recommendations

    def _calculate_trend(self, values: list[float]) -> float:
        """Calculate trend direction using simple linear regression."""
        if len(values) < 2:
            return 0.0
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        slope = numerator / denominator

        return max(-1.0, min(1.0, slope * 5))


class StrategyDriftDetector:
    """Detects when strategy performance degrades significantly."""

    def __init__(
        self,
        window_size: int = 30,
        drift_threshold: float = 0.25,
        recovery_threshold: float = 0.15,
    ):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.recovery_threshold = recovery_threshold
        self._baseline: dict[str, float] = {}
        self._recent_performance: list[dict[str, float]] = []
        self.logger = logging.getLogger(__name__)

    def set_baseline(self, win_rate: float, profit_factor: float) -> None:
        """Set baseline performance metrics."""
        self._baseline = {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
        }
        self.logger.info(f"Drift detector baseline set: WR={win_rate:.1f}%, PF={profit_factor:.2f}")

    def check_drift(self, current_wr: float, current_pf: float) -> DriftDetectionState:
        """Check if strategy has drifted from baseline."""
        if not self._baseline:
            return DriftDetectionState(baseline_win_rate=current_wr, baseline_profit_factor=current_pf)

        self._recent_performance.append({"win_rate": current_wr, "profit_factor": current_pf})
        if len(self._recent_performance) > self.window_size:
            self._recent_performance = self._recent_performance[-self.window_size:]

        wr_change = (current_wr - self._baseline["win_rate"]) / max(self._baseline["win_rate"], 1)
        pf_change = (current_pf - self._baseline["profit_factor"]) / max(self._baseline["profit_factor"], 0.1)

        drift_magnitude = abs(wr_change) + abs(pf_change)

        state = DriftDetectionState(
            baseline_win_rate=self._baseline["win_rate"],
            baseline_profit_factor=self._baseline["profit_factor"],
            drift_detected=drift_magnitude > self.drift_threshold,
            drift_magnitude=drift_magnitude,
        )

        if drift_magnitude > self.drift_threshold and not state.drift_start_time:
            state.drift_start_time = datetime.now(timezone.utc)
            self.logger.warning(f"Strategy drift detected: magnitude={drift_magnitude:.2f}")

        return state

    def should_adapt(self, drift_state: DriftDetectionState) -> bool:
        """Determine if adaptation should be applied."""
        if not drift_state.drift_detected:
            return False

        if drift_state.drift_magnitude > self.drift_threshold * 2:
            return True

        if drift_state.drift_start_time:
            drift_duration = datetime.now(timezone.utc) - drift_state.drift_start_time
            if drift_duration > timedelta(hours=2):
                return True

        return False


class RollingOptimizer:
    """Continuous optimization based on rolling windows."""

    def __init__(
        self,
        window_size: int = 100,
        min_improvement: float = 0.05,
    ):
        self.window_size = window_size
        self.min_improvement = min_improvement
        self._best_performance: float = float('-inf')
        self.logger = logging.getLogger(__name__)

    def should_optimize(self, recent_pnl: float, trade_count: int) -> bool:
        """Determine if optimization should run."""
        if trade_count < 20:
            return False

        if recent_pnl > self._best_performance + self.min_improvement:
            self._best_performance = recent_pnl
            return False

        if recent_pnl < self._best_performance - self.min_improvement:
            return True

        return False


class IntelligentAutoImprover:
    """
    Enhanced Auto-Improver with ML self-tuning, drift detection, and rolling optimization.
    """

    def __init__(self, enabled: bool = True):
        self.logger = logging.getLogger(__name__)
        self.enabled = enabled

        self._basic_auto_improver: AutoImprover | None = None
        self._advanced_auto_improver: AdvancedAutoImprover | None = None

        self._performance = TradePerformance()
        self._metrics = ImprovementMetrics()
        self._ml_stats = MLRunningStats()

        self._optimization_task: asyncio.Task | None = None
        self._self_tuning_task: asyncio.Task | None = None
        self._monitoring_task: asyncio.Task | None = None
        self._optimization_interval_hours = 6
        self._min_trades_before_optimization = 5

        self._loss_prevention_active = False
        self._consecutive_loss_threshold = 3
        self._emergency_optimization_triggered = False

        self._recent_trades: list[dict[str, Any]] = []
        self._max_recent_trades = 100

        # NEW: Advanced self-tuning components
        self._ml_tuner = MLBasedSelfTuner(history_window=50)
        self._drift_detector = StrategyDriftDetector()
        self._rolling_optimizer = RollingOptimizer()
        self._current_params = self._get_default_params()

        # NEW: Emergency shutdown thresholds
        self._max_consecutive_losses_for_shutdown = 15
        self._min_win_rate_for_shutdown = 15.0

        # NEW: Persistent state
        self._state_file = Path("data/auto_improver_state.pkl")
        self._load_state()

        if self.enabled:
            self._initialize_improvers()

    def _get_default_params(self) -> dict[str, Any]:
        return {
            "risk_per_trade": 0.01,
            "min_confidence": 0.70,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "max_holding_minutes": 60,
        }

    def _load_state(self) -> None:
        """Load persistent state from disk."""
        try:
            if self._state_file.exists():
                with open(self._state_file, "rb") as f:
                    state = pickle.load(f)
                    self._performance = state.get("performance", TradePerformance())
                    self._metrics = state.get("metrics", ImprovementMetrics())
                    self._current_params = state.get("params", self._get_default_params())
                    self._drift_detector.set_baseline(
                        state.get("baseline_wr", 50.0),
                        state.get("baseline_pf", 1.5)
                    )
                self.logger.info("Auto-improver state loaded from disk")
        except Exception as e:
            self.logger.warning(f"Could not load state: {e}")

    def _save_state(self) -> None:
        """Save state to disk for persistence."""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._state_file, "wb") as f:
                pickle.dump({
                    "performance": self._performance,
                    "metrics": self._metrics,
                    "params": self._current_params,
                    "baseline_wr": self._drift_detector._baseline.get("win_rate", 50.0),
                    "baseline_pf": self._drift_detector._baseline.get("profit_factor", 1.5),
                }, f)
        except Exception as e:
            self.logger.warning(f"Could not save state: {e}")

    def _initialize_improvers(self) -> None:
        try:
            basic_config = AutoImproverConfig()
            basic_config.auto_deploy = False
            self._basic_auto_improver = AutoImprover(config=basic_config)

            advanced_config = AdvancedImproverConfig()
            advanced_config.auto_deploy = False
            self._advanced_auto_improver = AdvancedAutoImprover(config=advanced_config)

            self.logger.info("Auto-improvement systems initialized (ENHANCED VERSION)")
        except Exception as exc:
            self.logger.error(f"Failed to initialize auto-improvement: {exc}")
            self.enabled = False

    async def start(self) -> None:
        if not self.enabled:
            self.logger.info("Auto-improvement disabled")
            return

        self.logger.info("Starting enhanced auto-improvement scheduler")

        # Set baseline if we have performance data
        if self._performance.total_trades > 0:
            self._drift_detector.set_baseline(
                self._performance.win_rate,
                self._performance.profit_factor
            )

        self._optimization_task = asyncio.create_task(self._optimization_loop())
        self._self_tuning_task = asyncio.create_task(self._self_tuning_loop())
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def stop(self) -> None:
        for task in [self._optimization_task, self._self_tuning_task, self._monitoring_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._save_state()
        self.logger.info("Auto-improvement scheduler stopped")

    async def _self_tuning_loop(self) -> None:
        """New: Continuous self-tuning using ML."""
        while True:
            try:
                await asyncio.sleep(300)

                if self._performance.total_trades < 10:
                    continue

                # Record current performance for ML
                self._ml_tuner.record_performance(
                    self._performance.win_rate,
                    self._performance.profit_factor,
                    self._current_params
                )

                # Get ML recommendations
                recommendations = self._ml_tuner.predict_optimal_params()

                if recommendations.get("action") in ["reduce_risk", "increase_aggression"]:
                    confidence = recommendations.get("confidence", 0.5)
                    if confidence > 0.7:
                        self._apply_ml_recommendation(recommendations)

                # Check for strategy drift
                drift_state = self._drift_detector.check_drift(
                    self._performance.win_rate,
                    self._performance.profit_factor
                )

                if drift_state.drift_detected:
                    self.logger.warning(f"Drift detected: {drift_state.drift_magnitude:.2f}")
                    if self._drift_detector.should_adapt(drift_state):
                        await self._handle_drift_adaptation(drift_state)

                # Check rolling optimization
                if self._rolling_optimizer.should_optimize(
                    self._performance.total_pnl,
                    self._performance.total_trades
                ):
                    self.logger.info("Triggering rolling optimization")
                    await self._run_optimization()

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.logger.error(f"Self-tuning loop error: {exc}")

    def _apply_ml_recommendation(self, recommendation: dict[str, Any]) -> None:
        """Apply ML-based parameter adjustments."""
        action = recommendation.get("action")

        if action == "reduce_risk":
            self._current_params["risk_per_trade"] = max(
                0.005,
                self._current_params["risk_per_trade"] * 0.8
            )
            self._current_params["stop_loss_pct"] = max(
                0.01,
                self._current_params["stop_loss_pct"] * 0.9
            )
            self.logger.info(f"ML applied risk reduction: {self._current_params}")

        elif action == "increase_aggression":
            self._current_params["risk_per_trade"] = min(
                0.02,
                self._current_params["risk_per_trade"] * 1.2
            )
            self.logger.info(f"ML increased aggression: {self._current_params}")

        self._save_state()

    async def _handle_drift_adaptation(self, drift_state: DriftDetectionState) -> None:
        """Handle strategy drift with targeted adaptation."""
        self.logger.warning(f"Applying drift adaptation for magnitude {drift_state.drift_magnitude:.2f}")

        # Reduce position size
        self._current_params["risk_per_trade"] *= 0.7
        self._current_params["min_confidence"] = max(
            0.8,
            self._current_params["min_confidence"] + 0.1
        )

        # Trigger emergency optimization
        await self._run_optimization()

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

    async def _monitoring_loop(self) -> None:
        """Monitor auto-improvement system health and log status."""
        while True:
            try:
                await asyncio.sleep(120)
                if self._performance.total_trades > 0:
                    self.logger.info(
                        f"Auto-Improver Status: trades={self._performance.total_trades}, "
                        f"wr={self._performance.win_rate:.1f}%, "
                        f"pf={self._performance.profit_factor:.2f}, "
                        f"losses={self._performance.consecutive_losses}, "
                        f"optimizations={self._metrics.optimization_count}"
                    )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.logger.error(f"Monitoring loop error: {exc}")
                await asyncio.sleep(60)

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

                    # Update baseline after successful optimization
                    self._drift_detector.set_baseline(
                        self._performance.win_rate,
                        self._performance.profit_factor
                    )

                    self.logger.info(f"Optimization successful: {result.get('details', {})}")
                else:
                    self._metrics.failed_optimizations += 1
                    self.logger.warning(f"Optimization failed: {result.get('error')}")

        except Exception as exc:
            self._metrics.failed_optimizations += 1
            self.logger.error(f"Optimization error: {exc}")
        finally:
            self._save_state()

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

        try:
            asyncio.get_running_loop().call_later(
                3600,
                lambda: setattr(self, "_emergency_optimization_triggered", False)
            )
        except RuntimeError:
            self._emergency_optimization_triggered = False

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
            "current_params": self._current_params,
            "ml_recommendations": self._ml_tuner.predict_optimal_params(),
        }

    def get_current_params(self) -> dict[str, Any]:
        """Return current operating parameters."""
        return self._current_params.copy()

    def should_block_trading(self) -> bool:
        if not self.enabled:
            return False

        if self._performance.max_consecutive_losses >= self._max_consecutive_losses_for_shutdown:
            self.logger.critical(f"Trading blocked: {self._performance.max_consecutive_losses} consecutive losses")
            return True

        if self._performance.win_rate < self._min_win_rate_for_shutdown and self._performance.total_trades >= 20:
            self.logger.critical(f"Trading blocked: win rate {self._performance.win_rate:.1f}% below threshold")
            return True

        return False