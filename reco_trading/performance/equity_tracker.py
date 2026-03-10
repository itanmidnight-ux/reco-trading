from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class EquityTracker:
    """Tracks equity curve and drawdown in real time."""

    points: list[float] = field(default_factory=list)

    def add_point(self, equity: float) -> None:
        self.points.append(float(equity))

    def current_drawdown(self) -> float:
        if not self.points:
            return 0.0
        peak = max(self.points)
        if peak <= 0:
            return 0.0
        return max((peak - self.points[-1]) / peak, 0.0)

    def max_drawdown(self) -> float:
        if not self.points:
            return 0.0
        peak = self.points[0]
        max_dd = 0.0
        for value in self.points:
            peak = max(peak, value)
            if peak > 0:
                max_dd = max(max_dd, (peak - value) / peak)
        return max_dd
