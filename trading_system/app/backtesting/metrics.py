from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BacktestMetrics:
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    profit_factor: float
    expectancy: float
