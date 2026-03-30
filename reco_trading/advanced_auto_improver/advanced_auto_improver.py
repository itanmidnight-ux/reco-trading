"""
Advanced Auto-Improver Orchestrator.
Combines all advanced modules for autonomous optimization.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from reco_trading.advanced_auto_improver.experimentation_engine import ABTestRunner, ExperimentationEngine
from reco_trading.advanced_auto_improver.failure_detection_system import FailureDetector, FailureResponseManager
from reco_trading.advanced_auto_improver.market_regime_detector import MarketRegime, MarketRegimeDetector
from reco_trading.advanced_auto_improver.meta_strategy_engine import MetaStrategyEngine
from reco_trading.advanced_auto_improver.monte_carlo_simulator import MonteCarloSimulator
from reco_trading.advanced_auto_improver.overfitting_detector import OverfittingDetector
from reco_trading.advanced_auto_improver.risk_adaptation_engine import RiskAdaptationEngine
from reco_trading.advanced_auto_improver.self_evaluation_engine import SelfEvaluationEngine
from reco_trading.advanced_auto_improver.strategy_versioning import StrategyVersioning
from reco_trading.advanced_auto_improver.walk_forward_optimizer import WalkForwardOptimizer
from reco_trading.auto_improver.auto_improver import AutoImprover
from reco_trading.auto_improver.strategy_generator import StrategyGenerator
from reco_trading.auto_improver.evaluator_engine import EvaluatorEngine
from reco_trading.auto_improver.strategy_selector import StrategySelector

logger = logging.getLogger(__name__)


@dataclass
class AdvancedImproverConfig:
    """Configuration for advanced auto-improver."""
    enable_market_regime_detection: bool = True
    enable_meta_strategy: bool = True
    enable_overfitting_detection: bool = True
    enable_walk_forward: bool = True
    enable_monte_carlo: bool = True
    enable_risk_adaptation: bool = True
    enable_self_evaluation: bool = True
    enable_failure_detection: bool = True
    enable_versioning: bool = True
    enable_experimentation: bool = True
    
    max_consecutive_losses: int = 7
    max_drawdown: float = 30.0
    
    walk_forward_train_days: int = 60
    walk_forward_validation_days: int = 14
    
    monte_carlo_simulations: int = 1000


class AdvancedAutoImprover:
    """
    Advanced Auto-Improver Orchestrator.
    
    Combines:
    - Market Regime Detection
    - Meta-Strategy Engine
    - Overfitting Detection
    - Walk-Forward Optimization
    - Monte Carlo Simulation
    - Risk Adaptation
    - Self-Evaluation
    - Failure Detection
    - Strategy Versioning
    - Experimentation
    """

    def __init__(self, config: Optional[AdvancedImproverConfig] = None):
        self.config = config or AdvancedImproverConfig()
        
        self.market_regime_detector = MarketRegimeDetector()
        self.meta_strategy_engine = MetaStrategyEngine()
        self.overfitting_detector = OverfittingDetector()
        self.walk_forward_optimizer = WalkForwardOptimizer(
            train_days=self.config.walk_forward_train_days,
            validation_days=self.config.walk_forward_validation_days,
        )
        self.monte_carlo_simulator = MonteCarloSimulator(
            num_simulations=self.config.monte_carlo_simulations,
        )
        self.risk_adaptation_engine = RiskAdaptationEngine()
        self.self_evaluation_engine = SelfEvaluationEngine()
        self.failure_detector = FailureDetector(
            max_consecutive_losses=self.config.max_consecutive_losses,
            max_drawdown=self.config.max_drawdown,
        )
        self.failure_response = FailureResponseManager(self.failure_detector)
        self.strategy_versioning = StrategyVersioning()
        self.experimentation_engine = ExperimentationEngine()
        
        self.base_auto_improver = AutoImprover()
        self.strategy_generator = StrategyGenerator()
        self.evaluator = EvaluatorEngine()
        self.selector = StrategySelector()
        
        self._running = False

    def update_market_data(self, price: float, volume: float) -> None:
        """Update market data for regime detection."""
        if self.config.enable_market_regime_detection:
            regime_metrics = self.market_regime_detector.add_data_point(price, volume)
            
            if self.config.enable_meta_strategy:
                self.meta_strategy_engine.adjust_for_regime(regime_metrics.regime)
        
        if self.config.enable_risk_adaptation:
            self.risk_adaptation_engine.update_market_data(
                current_balance=10000,
                current_drawdown=self.failure_detector._current_drawdown,
                recent_pnl=0,
            )

    def record_trade(self, pnl: float, is_error: bool = False) -> None:
        """Record trade for evaluation."""
        self.self_evaluation_engine.record_trade({"pnl": pnl})
        
        failure = self.failure_detector.check_trade_result(pnl, is_error)
        
        if failure:
            response = self.failure_response.execute_response(failure)
            logger.warning(f"Failure response: {response}")

    def should_trade(self) -> tuple[bool, str]:
        """Determine if trading should be allowed."""
        is_paused, reason = self.failure_response.is_paused()
        if is_paused:
            return False, f"Trading paused: {reason}"
        
        should_pause, pause_reason = self.failure_response.should_pause()
        if should_pause:
            return False, f"Should pause: {pause_reason}"
        
        if self.config.enable_market_regime_detection:
            should_trade, trade_reason = self.market_regime_detector.should_trade()
            if not should_trade:
                return False, trade_reason
        
        if self.config.enable_self_evaluation:
            evaluation = self.self_evaluation_engine.evaluate()
            if evaluation.should_switch_strategy:
                return False, "Strategy performance degraded"
        
        return True, "Trading allowed"

    def get_current_risk_parameters(self) -> dict[str, float]:
        """Get current risk parameters."""
        regime = None
        
        if self.config.enable_market_regime_detection:
            current_regime = self.market_regime_detector.get_current_regime()
            if current_regime:
                regime = current_regime.regime
        
        volatility = 2.0
        if regime:
            current_regime = self.market_regime_detector.get_current_regime()
            if current_regime:
                volatility = current_regime.volatility
        
        risk_params = self.risk_adaptation_engine.calculate_adapted_parameters(
            regime=regime,
            volatility=volatility,
        )
        
        return {
            "position_size": risk_params.position_size,
            "stop_loss": risk_params.stop_loss,
            "take_profit": risk_params.take_profit,
        }

    def analyze_strategy_robustness(
        self,
        strategy_params: dict[str, Any],
        historical_trades: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Analyze strategy robustness."""
        results = {}
        
        if self.config.enable_overfitting_detection and len(historical_trades) >= 30:
            train_size = len(historical_trades) // 2
            train_data = historical_trades[:train_size]
            val_data = historical_trades[train_size:]
            
            overfitting = self.overfitting_detector.analyze_rolling_window(
                [{"roi": sum(t.get("pnl", 0) for t in train_data)}],
                [{"roi": sum(t.get("pnl", 0) for t in val_data)}],
            )
            
            results["overfitting"] = self.overfitting_detector.to_dict()
        
        if self.config.enable_monte_carlo and historical_trades:
            mc_analysis = self.monte_carlo_simulator.simulate(historical_trades)
            results["monte_carlo"] = self.monte_carlo_simulator.to_dict(mc_analysis)
            
            robust, reason = self.monte_carlo_simulator.is_robust(mc_analysis)
            results["is_robust"] = robust
            results["robustness_reason"] = reason
        
        return results

    def evaluate_current_state(self) -> dict[str, Any]:
        """Evaluate current system state."""
        state = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        if self.config.enable_market_regime_detection:
            state["market_regime"] = self.market_regime_detector.to_dict()
        
        if self.config.enable_self_evaluation:
            state["self_evaluation"] = self.self_evaluation_engine.to_dict()
        
        if self.config.enable_failure_detection:
            state["failure_detection"] = self.failure_detector.to_dict()
        
        if self.config.enable_risk_adaptation:
            state["risk_adaptation"] = self.risk_adaptation_engine.to_dict()
        
        if self.config.enable_meta_strategy:
            state["meta_strategy"] = self.meta_strategy_engine.to_dict()
        
        state["should_trade"] = self.should_trade()
        state["risk_parameters"] = self.get_current_risk_parameters()
        
        return state

    def create_strategy_version(
        self,
        strategy_id: str,
        parameters: dict[str, Any],
        metrics: dict[str, float],
    ) -> str:
        """Create a new strategy version."""
        version = self.strategy_versioning.create_version(
            strategy_id=strategy_id,
            parameters=parameters,
            metrics=metrics,
        )
        return version.version_id

    def rollback_to_version(self, version_id: str) -> bool:
        """Rollback to a specific version."""
        return self.strategy_versioning.rollback_to_version(version_id)

    def start_experiment(
        self,
        name: str,
        control_params: dict[str, Any],
        variant_params: dict[str, Any],
    ) -> tuple[str, str]:
        """Start an A/B test experiment."""
        ab_runner = ABTestRunner(self.experimentation_engine)
        control, variant = ab_runner.start_ab_test(name, control_params, variant_params)
        return control.experiment_id, variant.experiment_id

    def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "system_state": self.evaluate_current_state(),
            "experiments": self.experimentation_engine.to_dict(),
            "config": {
                "market_regime_detection": self.config.enable_market_regime_detection,
                "meta_strategy": self.config.enable_meta_strategy,
                "overfitting_detection": self.config.enable_overfitting_detection,
                "monte_carlo": self.config.enable_monte_carlo,
                "risk_adaptation": self.config.enable_risk_adaptation,
                "self_evaluation": self.config.enable_self_evaluation,
                "failure_detection": self.config.enable_failure_detection,
            },
        }
