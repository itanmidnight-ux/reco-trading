from __future__ import annotations

import numpy as np


def sharpe(returns, periods=365*24*12):
    r = np.asarray(returns)
    return np.sqrt(periods) * r.mean() / (r.std() + 1e-12)


def sortino(returns, periods=365*24*12):
    r = np.asarray(returns)
    downside = r[r < 0]
    return np.sqrt(periods) * r.mean() / (downside.std() + 1e-12)


def max_drawdown(equity_curve):
    ec = np.asarray(equity_curve)
    peak = np.maximum.accumulate(ec)
    dd = (ec - peak) / peak
    return float(dd.min())


def calmar(returns, equity_curve):
    mdd = abs(max_drawdown(equity_curve)) + 1e-12
    return float(np.mean(returns) / mdd)


def profit_factor(returns):
    r = np.asarray(returns)
    gain = r[r > 0].sum()
    loss = abs(r[r < 0].sum()) + 1e-12
    return float(gain / loss)
