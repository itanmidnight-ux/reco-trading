from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


class ConditionalPerformanceTracker:
    def __init__(self, window: int = 120) -> None:
        self.window = max(int(window), 20)
        self._pnls: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=self.window))
        self._returns: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=self.window))

    def record_trade(self, regime: str, pnl: float, trade_return: float) -> None:
        key = str(regime or 'UNKNOWN')
        pnl_value = float(pnl) if np.isfinite(pnl) else 0.0
        ret_value = float(trade_return) if np.isfinite(trade_return) else 0.0
        self._pnls[key].append(pnl_value)
        self._returns[key].append(ret_value)

    def summary(self, regime: str) -> dict[str, float]:
        key = str(regime or 'UNKNOWN')
        arr = np.asarray(self._pnls.get(key, []), dtype=float)
        rets = np.asarray(self._returns.get(key, []), dtype=float)
        if arr.size == 0:
            return {'expectancy': 0.0, 'win_rate': 0.0, 'sharpe': 0.0, 'trades': 0.0}

        wins = arr[arr > 0.0]
        std = float(rets.std(ddof=1) if rets.size > 1 else 0.0)
        sharpe = float((rets.mean() / std) * np.sqrt(min(rets.size, 252))) if std > 0 else 0.0
        return {
            'expectancy': float(arr.mean()),
            'win_rate': float(wins.size / arr.size),
            'sharpe': sharpe,
            'trades': float(arr.size),
        }
