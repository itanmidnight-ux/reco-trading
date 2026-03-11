from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd

from reco_trading.backtesting.simulator import SimulatedTrade


@dataclass(slots=True)
class PerformanceMetrics:
    total_return: float
    win_rate: float
    max_drawdown: float
    profit_factor: float
    sharpe_ratio: float
    expectancy: float


def compute_metrics(trades: list[SimulatedTrade], starting_equity: float, equity_curve: list[float]) -> PerformanceMetrics:
    if starting_equity <= 0:
        starting_equity = 1.0

    closed = [t for t in trades if t.exit_price is not None]
    total_pnl = sum(t.pnl for t in closed)
    total_return = total_pnl / starting_equity

    wins = [t.pnl for t in closed if t.pnl > 0]
    losses = [t.pnl for t in closed if t.pnl < 0]
    win_rate = (len(wins) / len(closed)) if closed else 0.0

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    expectancy = (sum(t.pnl for t in closed) / len(closed)) if closed else 0.0

    curve = pd.Series(equity_curve if equity_curve else [starting_equity], dtype=float)
    peaks = curve.cummax()
    drawdowns = (curve - peaks) / peaks.replace(0, 1)
    max_drawdown = abs(float(drawdowns.min()))

    returns = curve.pct_change().dropna()
    if returns.empty or returns.std(ddof=0) == 0:
        sharpe = 0.0
    else:
        sharpe = float((returns.mean() / returns.std(ddof=0)) * math.sqrt(252))

    return PerformanceMetrics(
        total_return=float(total_return),
        win_rate=float(win_rate),
        max_drawdown=float(max_drawdown),
        profit_factor=float(profit_factor),
        sharpe_ratio=float(sharpe),
        expectancy=float(expectancy),
    )
