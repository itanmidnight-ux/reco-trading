from __future__ import annotations

import math
from datetime import UTC, datetime


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def floor_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.floor(value / step) * step


def to_bps(value: float) -> float:
    return value * 10_000


__all__ = ['utc_now', 'floor_step', 'to_bps']
