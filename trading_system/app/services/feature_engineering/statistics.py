from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import fmean
from collections.abc import Sequence


@dataclass
class StatisticalConfirmation:
    t_stat: float
    trend_pseudo_pvalue: float
    hurst_proxy: float
    stationarity_proxy: float
    confidence: float


def _returns(prices: Sequence[float]) -> list[float]:
    if len(prices) < 3:
        return []
    return [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices)) if prices[i - 1] != 0]


def one_sample_t_stat(values: Sequence[float]) -> float:
    n = len(values)
    if n < 3:
        return 0.0
    mean = fmean(values)
    var = fmean([(x - mean) ** 2 for x in values])
    std = sqrt(var) if var > 0 else 0.0
    if std == 0:
        return 0.0
    return mean / (std / sqrt(n))


def trend_pseudo_pvalue(t_stat: float) -> float:
    # Aproximación simple y estable sin SciPy: |t| alto => p pequeño.
    at = abs(t_stat)
    return max(0.001, min(1.0, 1.0 / (1.0 + at * at)))


def hurst_proxy(prices: Sequence[float]) -> float:
    if len(prices) < 60:
        return 0.5
    short = _returns(prices[-30:])
    long = _returns(prices[-60:])
    if not short or not long:
        return 0.5
    vol_short = sqrt(fmean([x * x for x in short]))
    vol_long = sqrt(fmean([x * x for x in long]))
    if vol_short == 0:
        return 0.5
    ratio = vol_long / vol_short
    # ratio > sqrt(2) sugiere persistencia (proxy H>0.5)
    return max(0.1, min(0.9, 0.5 + (ratio - 1.414) * 0.2))


def stationarity_proxy(prices: Sequence[float]) -> float:
    if len(prices) < 80:
        return 0.5
    first = prices[-80:-40]
    second = prices[-40:]
    m1 = fmean(first)
    m2 = fmean(second)
    if m1 == 0:
        return 0.5
    drift = abs(m2 - m1) / abs(m1)
    # menor drift => más estacionario
    return max(0.0, min(1.0, 1 - min(1.0, drift * 8)))


def statistical_confirmation(prices: Sequence[float]) -> StatisticalConfirmation:
    rets = _returns(prices[-120:])
    t_stat = one_sample_t_stat(rets)
    pvalue = trend_pseudo_pvalue(t_stat)
    hurst = hurst_proxy(prices)
    stationarity = stationarity_proxy(prices)

    # confianza compuesta: tendencia estadísticamente significativa + persistencia + baja no-estacionariedad
    confidence = max(0.0, min(1.0, (1 - pvalue) * 0.45 + hurst * 0.35 + stationarity * 0.20))
    return StatisticalConfirmation(
        t_stat=t_stat,
        trend_pseudo_pvalue=pvalue,
        hurst_proxy=hurst,
        stationarity_proxy=stationarity,
        confidence=confidence,
    )
