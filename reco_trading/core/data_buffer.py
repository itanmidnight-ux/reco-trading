from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    returns: np.ndarray
    volatility: float
    atr: float
    trend_strength: float
    spread: float
    vwap_distance: float = 0.0
    bollinger_deviation: float = 0.0
    vwap_distance: float
    bollinger_deviation: float


@dataclass(frozen=True, slots=True)
class LearningStats:
    mean_return: float
    return_std: float
    rolling_volatility: float
    atr: float
    average_spread: float
    dominant_regime: str


class DataBuffer:
    def __init__(self, window_seconds: int = 300, max_spreads: int = 600) -> None:
        self.window_seconds = max(int(window_seconds), 60)
        self._spreads: deque[float] = deque(maxlen=max_spreads)
        self._ohlcv_frame = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    def push_ohlcv(self, frame: pd.DataFrame) -> None:
        merged = pd.concat([self._ohlcv_frame, frame], ignore_index=True)
        if 'timestamp' in merged.columns:
            merged = merged.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')
            if not merged.empty:
                cutoff = merged['timestamp'].max() - pd.Timedelta(seconds=self.window_seconds)
                merged = merged.loc[merged['timestamp'] >= cutoff]
        self._ohlcv_frame = merged.reset_index(drop=True)

    @property
    def ohlcv(self) -> pd.DataFrame:
        return self._ohlcv_frame.copy()

    def record_spread(self, spread: float) -> None:
        if np.isfinite(spread) and spread >= 0.0:
            self._spreads.append(float(spread))

    def learning_progress(self, first_timestamp_ms: int | None, now_ts: float) -> tuple[float, float]:
        if first_timestamp_ms is None:
            return 0.0, float(self.window_seconds)
        elapsed = max(now_ts - (float(first_timestamp_ms) / 1000.0), 0.0)
        remaining = max(self.window_seconds - elapsed, 0.0)
        completion = min(elapsed / self.window_seconds, 1.0)
        return completion, remaining

    def in_learning_phase(self, first_timestamp_ms: int | None, now_ts: float) -> bool:
        _, remaining = self.learning_progress(first_timestamp_ms, now_ts)
        return remaining > 0.0

    def build_snapshot(self, frame: pd.DataFrame | None = None) -> MarketSnapshot:
        source = self._ohlcv_frame if frame is None else frame
        if source.empty:
            return MarketSnapshot(returns=np.array([], dtype=float), volatility=0.0, atr=0.0, trend_strength=0.0, spread=0.0)

        close = source['close'].astype(float)
        high = source['high'].astype(float)
        low = source['low'].astype(float)
        volume = source['volume'].astype(float)
    def in_learning_phase(self, first_timestamp_ms: int | None, now_ts: float) -> bool:
        if first_timestamp_ms is None:
            return True
        elapsed = now_ts - (float(first_timestamp_ms) / 1000.0)
        return elapsed < self.window_seconds

    def build_snapshot(self, frame: pd.DataFrame) -> MarketSnapshot:
        close = frame['close'].astype(float)
        high = frame['high'].astype(float)
        low = frame['low'].astype(float)
        volume = frame['volume'].astype(float)

        returns = np.log(close / close.shift(1)).dropna().tail(300).to_numpy(dtype=float)
        rolling_volatility = float(pd.Series(returns).tail(40).std() or 0.0)

        true_range = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = float(true_range.rolling(14).mean().iloc[-1] or 0.0)

        ema12 = close.ewm(span=12, adjust=False).mean().iloc[-1]
        ema26 = close.ewm(span=26, adjust=False).mean().iloc[-1]
        trend_strength = float(abs(ema12 - ema26) / max(close.iloc[-1], 1e-9))

        avg_spread = float(np.mean(self._spreads)) if self._spreads else 0.0
        vwap = float((close * volume).rolling(30).sum().iloc[-1] / max(volume.rolling(30).sum().iloc[-1], 1e-9))
        vwap_distance = float((close.iloc[-1] - vwap) / max(vwap, 1e-9))
        mean20 = float(close.rolling(20).mean().iloc[-1])
        std20 = float(close.rolling(20).std().iloc[-1] or 0.0)
        bollinger_deviation = float((close.iloc[-1] - mean20) / max(2.0 * std20, 1e-9))
        true_range = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = float(true_range.rolling(14).mean().iloc[-1] or 0.0)

        trend_strength = float(abs(close.ewm(span=12, adjust=False).mean().iloc[-1] - close.ewm(span=26, adjust=False).mean().iloc[-1]) / max(close.iloc[-1], 1e-9))
        avg_spread = float(np.mean(self._spreads)) if self._spreads else 0.0

        vwap = float((close * volume).rolling(30).sum().iloc[-1] / max(volume.rolling(30).sum().iloc[-1], 1e-9))
        vwap_distance = float((close.iloc[-1] - vwap) / max(vwap, 1e-9))

        mean20 = float(close.rolling(20).mean().iloc[-1])
        std20 = float(close.rolling(20).std().iloc[-1] or 0.0)
        bollinger_deviation = float((close.iloc[-1] - mean20) / max(2 * std20, 1e-9))

        return MarketSnapshot(
            returns=returns.copy(),
            volatility=max(rolling_volatility, 0.0),
            atr=max(atr, 0.0),
            trend_strength=max(trend_strength, 0.0),
            spread=max(avg_spread, 0.0),
            vwap_distance=vwap_distance,
            bollinger_deviation=bollinger_deviation,
        )

    def learning_stats(self, frame: pd.DataFrame | None = None) -> LearningStats:
    def learning_stats(self, frame: pd.DataFrame) -> LearningStats:
        snapshot = self.build_snapshot(frame)
        returns = snapshot.returns
        mean_return = float(np.mean(returns)) if returns.size else 0.0
        return_std = float(np.std(returns)) if returns.size else 0.0

        if snapshot.volatility >= 0.008:
            regime = 'HIGH_VOL'
        elif snapshot.trend_strength >= 0.0025:
            regime = 'TREND'
        else:
            regime = 'RANGE'
        if snapshot.volatility < 0.003:
            regime = 'LOW_VOL'
        elif snapshot.trend_strength > 0.002:
            regime = 'TREND'
        else:
            regime = 'CHOPPY'

        return LearningStats(
            mean_return=mean_return,
            return_std=return_std,
            rolling_volatility=snapshot.volatility,
            atr=snapshot.atr,
            average_spread=snapshot.spread,
            dominant_regime=regime,
        )
