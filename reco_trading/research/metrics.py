from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew


def sharpe(returns, periods: int = 365 * 24 * 12):
    r = np.asarray(returns, dtype=float)
    return float(np.sqrt(periods) * r.mean() / (r.std() + 1e-12))


def sortino(returns, periods: int = 365 * 24 * 12):
    r = np.asarray(returns, dtype=float)
    downside = r[r < 0]
    return float(np.sqrt(periods) * r.mean() / (downside.std() + 1e-12))


def information_ratio(returns, benchmark_returns, periods: int = 365 * 24 * 12):
    r = np.asarray(returns, dtype=float)
    b = np.asarray(benchmark_returns, dtype=float)
    n = min(r.size, b.size)
    if n == 0:
        return 0.0
    active = r[-n:] - b[-n:]
    return float(np.sqrt(periods) * active.mean() / (active.std() + 1e-12))


def cvar(returns, alpha: float = 0.95):
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    q = np.quantile(r, 1.0 - alpha)
    tail = r[r <= q]
    return float(abs(tail.mean()) if tail.size else abs(q))


def max_drawdown(equity_curve):
    ec = np.asarray(equity_curve, dtype=float)
    peak = np.maximum.accumulate(ec)
    dd = (ec - peak) / (peak + 1e-12)
    return float(dd.min())


def calmar(returns, equity_curve, periods: int = 365 * 24 * 12):
    annual_return = float(np.mean(returns) * periods)
    mdd = abs(max_drawdown(equity_curve)) + 1e-12
    return annual_return / mdd


def ulcer_index(equity_curve):
    ec = np.asarray(equity_curve, dtype=float)
    peak = np.maximum.accumulate(ec)
    drawdown_pct = ((ec / (peak + 1e-12)) - 1.0) * 100.0
    return float(np.sqrt(np.mean(np.square(np.minimum(drawdown_pct, 0.0)))))


def equity_curve_smoothness(equity_curve):
    ec = np.asarray(equity_curve, dtype=float)
    if ec.size < 3:
        return 0.0
    second_diff = np.diff(ec, n=2)
    return float(1.0 / (1.0 + np.mean(np.abs(second_diff))))


def return_skewness(returns):
    r = np.asarray(returns, dtype=float)
    return float(skew(r, bias=False)) if r.size > 2 else 0.0


def return_kurtosis(returns):
    r = np.asarray(returns, dtype=float)
    return float(kurtosis(r, fisher=True, bias=False)) if r.size > 3 else 0.0


def expectancy(returns):
    r = np.asarray(returns, dtype=float)
    wins = r[r > 0]
    losses = r[r < 0]
    if r.size == 0:
        return 0.0
    win_rate = wins.size / r.size
    loss_rate = losses.size / r.size
    avg_win = wins.mean() if wins.size else 0.0
    avg_loss = abs(losses.mean()) if losses.size else 0.0
    return float((win_rate * avg_win) - (loss_rate * avg_loss))


def profit_factor(returns):
    r = np.asarray(returns, dtype=float)
    gain = r[r > 0].sum()
    loss = abs(r[r < 0].sum()) + 1e-12
    return float(gain / loss)
