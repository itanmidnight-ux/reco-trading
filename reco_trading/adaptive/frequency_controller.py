from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone

import numpy as np


class FrequencyController:
    def __init__(self, target_trades_per_day: int = 10, adjustment_strength: float = 0.05) -> None:
        self.target_trades_per_day = int(max(target_trades_per_day, 1))
        self.adjustment_strength = float(np.clip(adjustment_strength, 0.001, 0.5))
        self.trade_timestamps: deque[datetime] = deque()
        self.window = timedelta(days=1)

    def _prune(self, now: datetime) -> None:
        cutoff = now - self.window
        while self.trade_timestamps and self.trade_timestamps[0] < cutoff:
            self.trade_timestamps.popleft()

    def register_trade(self, timestamp: datetime | None = None) -> None:
        now = timestamp or datetime.now(timezone.utc)
        self.trade_timestamps.append(now)
        self._prune(now)

    def adjust_threshold(self, dynamic_edge_threshold: float, friction_cost: float, now: datetime | None = None) -> float:
        ref_now = now or datetime.now(timezone.utc)
        self._prune(ref_now)
        threshold = float(max(dynamic_edge_threshold, 0.0))
        trade_count = len(self.trade_timestamps)

        if trade_count < self.target_trades_per_day:
            threshold *= (1.0 - self.adjustment_strength)
        elif trade_count > self.target_trades_per_day:
            threshold *= (1.0 + self.adjustment_strength)

        return float(max(threshold, max(float(friction_cost), 0.0)))
