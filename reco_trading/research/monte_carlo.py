from __future__ import annotations

import numpy as np

from reco_trading.research.metrics import max_drawdown


class MonteCarlo:
    def simulate(self, returns, iterations: int = 3000, seed: int = 42):
        rng = np.random.default_rng(seed)
        r = np.asarray(returns, dtype=float)
        if r.size == 0:
            return {
                "p05_terminal_equity": 1.0,
                "p50_terminal_equity": 1.0,
                "p95_terminal_equity": 1.0,
                "p95_drawdown": 0.0,
            }

        terminal = np.empty(iterations, dtype=float)
        drawdowns = np.empty(iterations, dtype=float)
        for i in range(iterations):
            sampled = rng.choice(r, size=r.size, replace=True)
            eq = np.cumprod(1.0 + sampled)
            terminal[i] = float(eq[-1])
            drawdowns[i] = abs(max_drawdown(eq))

        return {
            "p05_terminal_equity": float(np.quantile(terminal, 0.05)),
            "p50_terminal_equity": float(np.quantile(terminal, 0.50)),
            "p95_terminal_equity": float(np.quantile(terminal, 0.95)),
            "p95_drawdown": float(np.quantile(drawdowns, 0.95)),
        }
