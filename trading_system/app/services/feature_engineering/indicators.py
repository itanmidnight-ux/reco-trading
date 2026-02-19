from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean, pstdev


def sma(values: Sequence[float], period: int) -> float:
    return 0.0 if len(values) < period else fmean(values[-period:])


def ema(values: Sequence[float], period: int) -> float:
    if len(values) < period:
        return 0.0
    alpha = 2 / (period + 1)
    e = values[-period]
    for v in values[-period + 1 :]:
        e = alpha * v + (1 - alpha) * e
    return e


def rsi(values: Sequence[float], period: int = 14) -> float:
    if len(values) < period + 1:
        return 50.0
    gains, losses = 0.0, 0.0
    w = values[-(period + 1) :]
    for a, b in zip(w, w[1:]):
        d = b - a
        gains += max(d, 0)
        losses += max(-d, 0)
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100 - 100 / (1 + rs)


def atr(high: Sequence[float], low: Sequence[float], close: Sequence[float], period: int = 14) -> float:
    if len(close) < period + 1:
        return 0.0
    tr = []
    for i in range(-period, 0):
        tr.append(max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1])))
    return fmean(tr)


def macd_hist(values: Sequence[float]) -> float:
    return ema(values, 12) - ema(values, 26) - ema([ema(values[:i], 12) - ema(values[:i], 26) for i in range(26, len(values)+1)], 9)


def bollinger_zscore(values: Sequence[float], period: int = 20) -> float:
    if len(values) < period:
        return 0.0
    m = fmean(values[-period:])
    s = pstdev(values[-period:]) or 1e-9
    return (values[-1] - m) / s


def rolling_vol(values: Sequence[float], period: int = 30) -> float:
    return 0.0 if len(values) < period else pstdev(values[-period:])


def zscore(values: Sequence[float], period: int = 50) -> float:
    if len(values) < period:
        return 0.0
    window = list(values[-period:])
    m = fmean(window)
    s = pstdev(window) or 1e-9
    return (window[-1] - m) / s


def skewness(values: Sequence[float], period: int = 50) -> float:
    if len(values) < period:
        return 0.0
    window = list(values[-period:])
    m = fmean(window)
    s = pstdev(window) or 1e-9
    return fmean([((x - m) / s) ** 3 for x in window])


def kurtosis(values: Sequence[float], period: int = 50) -> float:
    if len(values) < period:
        return 0.0
    window = list(values[-period:])
    m = fmean(window)
    s = pstdev(window) or 1e-9
    return fmean([((x - m) / s) ** 4 for x in window])
