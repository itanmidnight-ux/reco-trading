from __future__ import annotations

import random

from trading_system.app.backtesting.metrics import BacktestMetrics


class BacktestingSimulator:
    def run(self, trades: int = 300) -> BacktestMetrics:
        pnl = [random.gauss(0.001, 0.01) for _ in range(trades)]
        wins = [x for x in pnl if x > 0]
        losses = [x for x in pnl if x < 0]
        profit_factor = (sum(wins) / abs(sum(losses))) if losses else 99.0
        expectancy = sum(pnl) / len(pnl)
        max_dd = min(0.0, min(self._equity_curve(pnl)))
        sharpe = expectancy / (0.01 + 1e-9)
        return BacktestMetrics(sharpe=sharpe, sortino=sharpe * 1.1, max_drawdown=abs(max_dd), calmar=sharpe / max(0.01, abs(max_dd)), profit_factor=profit_factor, expectancy=expectancy)

    def _equity_curve(self, pnl: list[float]) -> list[float]:
        equity = 1.0
        peak = 1.0
        drawdowns = []
        for r in pnl:
            equity *= 1 + r
            peak = max(peak, equity)
            drawdowns.append((equity - peak) / peak)
        return drawdowns
