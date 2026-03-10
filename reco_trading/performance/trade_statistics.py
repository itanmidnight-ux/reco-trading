from __future__ import annotations

from dataclasses import dataclass, field


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
