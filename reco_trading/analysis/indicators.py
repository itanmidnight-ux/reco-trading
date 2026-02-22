from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True, slots=True)
class IndicatorSet:
    log_return: float
    ewma_vol_annualized: float
    atr: float
    return_zscore: float
    rsi14: float
    adx14: float
    spread_bps: float
    relative_volume_percentile: float


def _safe_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return frame[column].astype(float)


def compute_indicators(frame: pd.DataFrame, spread_bps: float, lookback: int = 200) -> IndicatorSet:
    source = frame.tail(max(lookback, 30)).copy()
    close = _safe_series(source, 'close')
    high = _safe_series(source, 'high')
    low = _safe_series(source, 'low')
    volume = _safe_series(source, 'volume')

    if close.size < 20:
        return IndicatorSet(0.0, 0.0, 0.0, 0.0, 50.0, 0.0, float(max(spread_bps, 0.0)), 0.5)

    log_returns = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    latest_log_return = float(log_returns.iloc[-1]) if not log_returns.empty else 0.0

    ewma_var = log_returns.ewm(span=20, adjust=False).var().iloc[-1] if not log_returns.empty else 0.0
    ewma_vol = float(np.sqrt(max(float(ewma_var or 0.0), 0.0)) * np.sqrt(TRADING_DAYS_PER_YEAR))

    prev_close = close.shift(1)
    true_range = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = float(true_range.rolling(14).mean().iloc[-1] or 0.0)

    mean_ret = float(log_returns.mean() or 0.0)
    std_ret = float(log_returns.std() or 0.0)
    zscore = 0.0 if std_ret <= 1e-9 else float((latest_log_return - mean_ret) / std_ret)

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0.0, np.nan)
    rsi = float((100.0 - (100.0 / (1.0 + rs))).fillna(50.0).iloc[-1])

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr14 = true_range.rolling(14).mean().replace(0.0, np.nan)
    plus_di = 100.0 * pd.Series(plus_dm, index=source.index).rolling(14).sum() / atr14
    minus_di = 100.0 * pd.Series(minus_dm, index=source.index).rolling(14).sum() / atr14
    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)).fillna(0.0)
    adx = float(dx.rolling(14).mean().fillna(0.0).iloc[-1])

    rel_vol = volume / volume.rolling(30).mean().replace(0.0, np.nan)
    rel_volume = float(rel_vol.rank(pct=True).fillna(0.5).iloc[-1])

    return IndicatorSet(
        log_return=latest_log_return,
        ewma_vol_annualized=max(ewma_vol, 0.0),
        atr=max(atr, 0.0),
        return_zscore=float(np.clip(zscore, -10.0, 10.0)),
        rsi14=float(np.clip(rsi, 0.0, 100.0)),
        adx14=max(adx, 0.0),
        spread_bps=float(max(spread_bps, 0.0)),
        relative_volume_percentile=float(np.clip(rel_volume, 0.0, 1.0)),
    )
