from __future__ import annotations

from dataclasses import dataclass

from reco_trading.performance.equity_tracker import EquityTracker
from reco_trading.performance.trade_statistics import TradeStatistics


@dataclass(slots=True)
class PerformanceSnapshot:
    win_rate: float
    average_profit: float
    average_loss: float
    profit_factor: float
    expectancy: float
    equity_curve: list[float]
    drawdown: float
    max_drawdown: float


class PerformanceMetricsCollector:
    """Collects and exposes real-time trading performance metrics."""

    def __init__(self) -> None:
        self.equity_tracker = EquityTracker()
        self.trade_stats = TradeStatistics()

    def on_trade_closed(self, pnl: float) -> None:
        self.trade_stats.record_trade(pnl)

    def on_equity_update(self, equity: float) -> None:
        self.equity_tracker.add_point(equity)

    def snapshot(self) -> PerformanceSnapshot:
        return PerformanceSnapshot(
            win_rate=self.trade_stats.win_rate,
            average_profit=self.trade_stats.average_profit,
            average_loss=self.trade_stats.average_loss,
            profit_factor=self.trade_stats.profit_factor,
            expectancy=self.trade_stats.expectancy,
            equity_curve=list(self.equity_tracker.points),
            drawdown=self.equity_tracker.current_drawdown(),
            max_drawdown=self.equity_tracker.max_drawdown(),
        )
