"""
Advanced Auto-Improver Module.
Advanced autonomous strategy optimization and adaptation system.
"""

from reco_trading.advanced_auto_improver.experimentation_engine import ABTestRunner, Experiment, ExperimentationEngine
from reco_trading.advanced_auto_improver.failure_detection_system import FailureDetector, FailureResponseManager, FailureEvent
from reco_trading.advanced_auto_improver.market_regime_detector import MarketRegime, MarketRegimeDetector, RegimeMetrics
from reco_trading.advanced_auto_improver.meta_strategy_engine import MetaStrategyEngine, StrategyAllocation
from reco_trading.advanced_auto_improver.monte_carlo_simulator import MonteCarloAnalysis, MonteCarloSimulator
from reco_trading.advanced_auto_improver.overfitting_detector import OverfittingDetector, OverfittingMetrics
from reco_trading.advanced_auto_improver.risk_adaptation_engine import RiskAdaptationEngine, RiskParameters
from reco_trading.advanced_auto_improver.self_evaluation_engine import EvaluationResult, HealthStatus, SelfEvaluationEngine
from reco_trading.advanced_auto_improver.strategy_versioning import StrategyVersion, StrategyVersioning
from reco_trading.advanced_auto_improver.walk_forward_optimizer import WalkForwardOptimizer, WalkForwardResult
from reco_trading.advanced_auto_improver.advanced_auto_improver import AdvancedAutoImprover, AdvancedImproverConfig

__all__ = [
    "AdvancedAutoImprover",
    "AdvancedImproverConfig",
    "MarketRegime",
    "MarketRegimeDetector",
    "RegimeMetrics",
    "MetaStrategyEngine",
    "StrategyAllocation",
    "OverfittingDetector",
    "OverfittingMetrics",
    "WalkForwardOptimizer",
    "WalkForwardResult",
    "MonteCarloSimulator",
    "MonteCarloAnalysis",
    "RiskAdaptationEngine",
    "RiskParameters",
    "SelfEvaluationEngine",
    "EvaluationResult",
    "HealthStatus",
    "FailureDetector",
    "FailureResponseManager",
    "FailureEvent",
    "StrategyVersioning",
    "StrategyVersion",
    "ExperimentationEngine",
    "Experiment",
    "ABTestRunner",
]
