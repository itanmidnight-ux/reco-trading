"""Backtesting package for offline simulation workflows."""

__all__ = [
    "BacktestReporter",
    "show_backtest_results",
    "store_backtest_results",
]

try:
    from .engine import BacktestEngine, BacktestResult
    __all__.extend(["BacktestEngine", "BacktestResult"])
except ImportError:
    pass

try:
    from .hyperopt import (
        HyperoptOptimizer,
        GridSearchOptimizer,
        HyperoptSpace,
        OptimizationResult,
        BestResult,
        OptimizerType,
        create_default_space,
    )
    __all__.extend([
        "HyperoptOptimizer",
        "GridSearchOptimizer", 
        "HyperoptSpace",
        "OptimizationResult",
        "BestResult",
        "OptimizerType",
        "create_default_space",
    ])
except ImportError:
    pass

from .reports import BacktestReporter, show_backtest_results, store_backtest_results
