from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    total_trades: int = 0
    total_profit: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0


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
            sharpe_ratio=self.trade_stats.sharpe_ratio,
            sortino_ratio=self.trade_stats.sortino_ratio,
            calmar_ratio=self.trade_stats.calmar_ratio,
            total_trades=self.trade_stats.total_trades,
            total_profit=self.trade_stats.total_profit,
            consecutive_wins=self.trade_stats.consecutive_wins,
            consecutive_losses=self.trade_stats.consecutive_losses,
        )

    def to_dict(self) -> dict[str, Any]:
        """Export all metrics as dictionary."""
        snapshot = self.snapshot()
        return {
            "win_rate": snapshot.win_rate,
            "average_profit": snapshot.average_profit,
            "average_loss": snapshot.average_loss,
            "profit_factor": snapshot.profit_factor,
            "expectancy": snapshot.expectancy,
            "sharpe_ratio": snapshot.sharpe_ratio,
            "sortino_ratio": snapshot.sortino_ratio,
            "calmar_ratio": snapshot.calmar_ratio,
            "max_drawdown": snapshot.max_drawdown,
            "total_trades": snapshot.total_trades,
            "total_profit": snapshot.total_profit,
            "consecutive_wins": snapshot.consecutive_wins,
            "consecutive_losses": snapshot.consecutive_losses,
            "equity_curve": snapshot.equity_curve,
        }


class DrawdownAnalyzer:
    """Analyzes drawdown periods and recovery times."""

    def __init__(self) -> None:
        self.equity_points: list[float] = []
        self.drawdown_periods: list[dict[str, Any]] = []

    def add_point(self, equity: float) -> None:
        self.equity_points.append(equity)
        self._analyze_drawdown()

    def _analyze_drawdown(self) -> None:
        if len(self.equity_points) < 2:
            return

        equity = self.equity_points[-1]
        peak = max(self.equity_points[:-1])
        
        if equity < peak:
            dd = (peak - equity) / peak * 100
            if not self.drawdown_periods or not self.drawdown_periods[-1].get("end"):
                self.drawdown_periods.append({
                    "start": len(self.equity_points) - 2,
                    "peak": peak,
                    "trough": equity,
                    "depth": dd,
                    "end": None,
                })
            else:
                self.drawdown_periods[-1]["trough"] = min(self.drawdown_periods[-1]["trough"], equity)
                self.drawdown_periods[-1]["depth"] = (self.drawdown_periods[-1]["peak"] - self.drawdown_periods[-1]["trough"]) / self.drawdown_periods[-1]["peak"] * 100
        elif self.drawdown_periods and not self.drawdown_periods[-1].get("end"):
            self.drawdown_periods[-1]["end"] = len(self.equity_points) - 1

    @property
    def average_drawdown(self) -> float:
        if not self.drawdown_periods:
            return 0.0
        return sum(dd["depth"] for dd in self.drawdown_periods) / len(self.drawdown_periods)

    @property
    def longest_drawdown(self) -> int:
        if not self.drawdown_periods:
            return 0
        return max((dd.get("end", 0) - dd["start"]) for dd in self.drawdown_periods if dd.get("end"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "average_drawdown": self.average_drawdown,
            "longest_drawdown_periods": self.longest_drawdown,
            "drawdown_count": len(self.drawdown_periods),
        }
