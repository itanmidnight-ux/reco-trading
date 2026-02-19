from __future__ import annotations

import numpy as np


class MonteCarlo:
    def simulate(self, returns, iterations: int = 2000, seed: int = 42):
        rng = np.random.default_rng(seed)
        r = np.asarray(returns, dtype=float)
        if r.size == 0:
            return {'p05_terminal_equity': 1.0, 'p50_terminal_equity': 1.0, 'p95_terminal_equity': 1.0}

        results = []
        for _ in range(iterations):
            sampled = rng.choice(r, size=r.size, replace=True)
            results.append(float(np.cumprod(1 + sampled)[-1]))

        return {
            'p05_terminal_equity': float(np.quantile(results, 0.05)),
            'p50_terminal_equity': float(np.quantile(results, 0.50)),
            'p95_terminal_equity': float(np.quantile(results, 0.95)),
        }
