from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TradeStatistics:
    """Aggregates PnL distributions and expectancy components."""

    pnls: list[float] = field(default_factory=list)

    def record_trade(self, pnl: float) -> None:
        self.pnls.append(float(pnl))

    @property
    def win_rate(self) -> float:
        if not self.pnls:
            return 0.0
        wins = sum(1 for p in self.pnls if p > 0)
        return wins / len(self.pnls)

    @property
    def average_profit(self) -> float:
        wins = [p for p in self.pnls if p > 0]
        return (sum(wins) / len(wins)) if wins else 0.0

    @property
    def average_loss(self) -> float:
        losses = [p for p in self.pnls if p < 0]
        return (sum(losses) / len(losses)) if losses else 0.0

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(p for p in self.pnls if p > 0)
        gross_loss = abs(sum(p for p in self.pnls if p < 0))
        return (gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    @property
    def expectancy(self) -> float:
        return (sum(self.pnls) / len(self.pnls)) if self.pnls else 0.0

    @property
    def total_trades(self) -> int:
        return len(self.pnls)

    @property
    def total_profit(self) -> float:
        return sum(self.pnls)

    @property
    def sharpe_ratio(self) -> float:
        """Calculate Sharpe Ratio (annualized)."""
        if len(self.pnls) < 2:
            return 0.0
        
        returns = [p / 100 for p in self.pnls]
        mean_return = sum(returns) / len(returns)
        
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance)
        
        if std_dev == 0:
            return 0.0
        
        sharpe = (mean_return / std_dev) * math.sqrt(252)
        return sharpe

    @property
    def sortino_ratio(self) -> float:
        """Calculate Sortino Ratio (annualized), uses downside deviation."""
        if len(self.pnls) < 2:
            return 0.0
        
        returns = [p / 100 for p in self.pnls]
        mean_return = sum(returns) / len(returns)
        
        downside_returns = [r for r in returns if r < 0]
        if not downside_returns:
            return float("inf") if mean_return > 0 else 0.0
        
        downside_variance = sum(r ** 2 for r in downside_returns) / len(downside_returns)
        downside_std = math.sqrt(downside_variance)
        
        if downside_std == 0:
            return 0.0
        
        sortino = (mean_return / downside_std) * math.sqrt(252)
        return sortino

    @property
    def calmar_ratio(self) -> float:
        """Calculate Calmar Ratio (annualized return / max drawdown)."""
        if not self.pnls:
            return 0.0
        
        total_return = sum(self.pnls) / 100
        max_dd = self.max_drawdown
        
        if max_dd == 0:
            return float("inf") if total_return > 0 else 0.0
        
        return total_return / max_dd

    @property
    def max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve."""
        if not self.pnls:
            return 0.0
        
        equity = 100.0
        peak = equity
        max_dd = 0.0
        
        for pnl in self.pnls:
            equity += pnl
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        return max_dd * 100

    @property
    def consecutive_wins(self) -> int:
        """Maximum consecutive winning trades."""
        if not self.pnls:
            return 0
        
        max_consecutive = 0
        current = 0
        
        for pnl in self.pnls:
            if pnl > 0:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 0
        
        return max_consecutive

    @property
    def consecutive_losses(self) -> int:
        """Maximum consecutive losing trades."""
        if not self.pnls:
            return 0
        
        max_consecutive = 0
        current = 0
        
        for pnl in self.pnls:
            if pnl < 0:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 0
        
        return max_consecutive

    def to_dict(self) -> dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "average_profit": self.average_profit,
            "average_loss": self.average_loss,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "total_profit": self.total_profit,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
        }
