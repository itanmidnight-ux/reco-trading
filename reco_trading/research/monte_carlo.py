from __future__ import annotations

import numpy as np


class MonteCarlo:
    def simulate(self, returns, iterations: int = 1000):
        r = np.asarray(returns)
        results = []
        for _ in range(iterations):
            shuffled = np.random.permutation(r)
            equity = np.cumprod(1 + shuffled)[-1]
            results.append(equity)
        return {
            'p05_terminal_equity': float(np.quantile(results, 0.05)),
            'p50_terminal_equity': float(np.quantile(results, 0.50)),
            'p95_terminal_equity': float(np.quantile(results, 0.95)),
        }
