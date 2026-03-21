from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(slots=True)
class SessionStats:
    total_trades: int
    win_rate: float
    current_streak: int
    recommendation: str
    profit_factor: float
    sharpe_estimate: float
    size_multiplier: float


class SessionTracker:
    def __init__(self, max_samples: int = 50) -> None:
        self.max_samples = max(int(max_samples), 5)
        self._recent_pnls: list[float] = []

    def record(self, pnl: float) -> None:
        self._recent_pnls.append(float(pnl))
        if len(self._recent_pnls) > self.max_samples:
            self._recent_pnls = self._recent_pnls[-self.max_samples:]

    @property
    def recent_pnls(self) -> list[float]:
        return list(self._recent_pnls)

    def stats(self) -> SessionStats:
        pnls = self._recent_pnls
        total = len(pnls)
        if total == 0:
            return SessionStats(0, 0.0, 0, 'NORMAL', 0.0, 0.0, 1.0)

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        win_rate = len(wins) / total if total else 0.0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)

        streak = 0
        for pnl in reversed(pnls):
            if pnl > 0:
                if streak < 0:
                    break
                streak += 1
            elif pnl < 0:
                if streak > 0:
                    break
                streak -= 1
            else:
                break

        mean = sum(pnls) / total
        if total > 1:
            variance = sum((p - mean) ** 2 for p in pnls) / (total - 1)
            sharpe = (mean / sqrt(variance)) if variance > 1e-12 else 0.0
        else:
            sharpe = 0.0

        recommendation = 'NORMAL'
        size_multiplier = 1.0
        if streak <= -3 or (total >= 5 and win_rate < 0.35 and mean < 0):
            recommendation = 'PAUSE'
            size_multiplier = 0.0
        elif streak <= -2 or (total >= 4 and win_rate < 0.45):
            recommendation = 'REDUCE_SIZE'
            size_multiplier = 0.6
        elif streak >= 3 and win_rate >= 0.60 and mean > 0:
            recommendation = 'INCREASE'
            size_multiplier = 1.1

        return SessionStats(total, win_rate, streak, recommendation, float(profit_factor), float(sharpe), size_multiplier)
