from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PatternSnapshot:
    engulfing: bool
    pin_bar: bool
    doji: bool
    breakout: bool
    rejection: bool


class PatternEngine:
    def detect(self, close: list[float], high: list[float], low: list[float]) -> PatternSnapshot:
        if len(close) < 4:
            return PatternSnapshot(False, False, False, False, False)
        body_last = abs(close[-1] - close[-2])
        range_last = max(1e-9, high[-1] - low[-1])
        doji = (body_last / range_last) < 0.1
        pin_bar = (body_last / range_last) < 0.25 and (close[-1] > close[-2])
        engulfing = (close[-1] > close[-2]) and (close[-2] < close[-3])
        breakout = close[-1] > max(close[-20:-1]) if len(close) > 20 else False
        rejection = close[-1] < high[-1] and close[-1] > low[-1] and body_last / range_last < 0.5
        return PatternSnapshot(engulfing, pin_bar, doji, breakout, rejection)
