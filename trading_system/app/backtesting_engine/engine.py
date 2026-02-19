from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import fmean
from typing import Callable


@dataclass
class BacktestReport:
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    profit_factor: float
    expectancy: float
    monte_carlo_p5: float
    stress_drawdown: float


class BacktestingEngine:
    def run(self, returns: list[float], fee_bps: float = 10, slippage_bps: float = 5, latency_ms: float = 75) -> BacktestReport:
        adjusted = [r - (fee_bps + slippage_bps) / 10000 for r in returns]
        latency_penalty = latency_ms / 1000 * 0.0001
        adjusted = [r - latency_penalty for r in adjusted]

        wins = [r for r in adjusted if r > 0]
        losses = [r for r in adjusted if r < 0]
        expectancy = fmean(adjusted) if adjusted else 0.0
        std = (fmean([(x - expectancy) ** 2 for x in adjusted]) ** 0.5) if adjusted else 1e-9
        downside = [x for x in adjusted if x < 0]
        downside_std = (fmean([x * x for x in downside]) ** 0.5) if downside else 1e-9
        sharpe = expectancy / (std + 1e-9)
        sortino = expectancy / (downside_std + 1e-9)

        dd = self._max_drawdown(adjusted)
        calmar = expectancy / max(dd, 1e-9)
        pf = (sum(wins) / abs(sum(losses))) if losses else 99.0

        mc = self._monte_carlo(adjusted, sims=200)
        stress_dd = self._stress_test(adjusted)

        return BacktestReport(sharpe, sortino, calmar, dd, pf, expectancy, mc, stress_dd)

    def walk_forward(self, returns: list[float], train: int = 200, test: int = 50) -> list[BacktestReport]:
        if train <= 0 or test <= 0:
            raise ValueError('train y test deben ser > 0')
        reports: list[BacktestReport] = []
        i = train
        while i + test <= len(returns):
            test_slice = returns[i : i + test]
            reports.append(self.run(test_slice))
            i += test
        return reports

    def simulate_live_loop(self, candles: list[float], strategy: Callable[[list[float]], float], warmup: int = 80) -> list[float]:
        realized: list[float] = []
        for idx in range(warmup, len(candles) - 1):
            history = candles[: idx + 1]
            signal = strategy(history)
            next_return = (candles[idx + 1] - candles[idx]) / max(candles[idx], 1e-9)
            realized.append(signal * next_return)
        return realized

    def _equity_curve(self, returns: list[float]) -> list[float]:
        eq = 1.0
        curve = []
        for r in returns:
            eq *= 1 + r
            curve.append(eq)
        return curve

    def _max_drawdown(self, returns: list[float]) -> float:
        curve = self._equity_curve(returns)
        peak = 1.0
        max_dd = 0.0
        for e in curve:
            peak = max(peak, e)
            dd = (peak - e) / peak
            max_dd = max(max_dd, dd)
        return max_dd

    def _monte_carlo(self, returns: list[float], sims: int) -> float:
        if not returns:
            return 0.0
        finals = []
        for _ in range(sims):
            sample = [random.choice(returns) for _ in range(len(returns))]
            curve = self._equity_curve(sample)
            finals.append(curve[-1] if curve else 1.0)
        finals.sort()
        return finals[max(0, int(0.05 * len(finals)) - 1)]

    def _stress_test(self, returns: list[float]) -> float:
        stressed = [r * 1.8 if r < 0 else r * 0.7 for r in returns]
        return self._max_drawdown(stressed)
