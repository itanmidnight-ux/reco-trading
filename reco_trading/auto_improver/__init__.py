"""
Auto-Improver Module for Reco-Trading.
Autonomous strategy optimization system.
"""

from reco_trading.auto_improver.auto_improver import AutoImprover, AutoImproverConfig, ImprovementCycle
from reco_trading.auto_improver.data_collector import DataCollector, DataSet, MarketDataPoint
from reco_trading.auto_improver.deployment_manager import DeploymentManager, DeploymentStatus
from reco_trading.auto_improver.evaluator_engine import BacktestEngine, EvaluationResult, EvaluatorEngine
from reco_trading.auto_improver.strategy_generator import IndicatorConfig, StrategyGenerator, StrategyVariant
from reco_trading.auto_improver.strategy_selector import SelectionResult, StrategySelector
from reco_trading.auto_improver.training_engine import SignalLabeler, TrainingEngine, TrainingResult

__all__ = [
    "AutoImprover",
    "AutoImproverConfig",
    "ImprovementCycle",
    "DataCollector",
    "DataSet",
    "MarketDataPoint",
    "DeploymentManager",
    "DeploymentStatus",
    "BacktestEngine",
    "EvaluationResult",
    "EvaluatorEngine",
    "IndicatorConfig",
    "StrategyGenerator",
    "StrategyVariant",
    "SelectionResult",
    "StrategySelector",
    "SignalLabeler",
    "TrainingEngine",
    "TrainingResult",
]
