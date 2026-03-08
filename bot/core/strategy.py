from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bot.config import BotConfig


@dataclass
class Signal:
    action: str
    confidence: float
    expected_move: float


class DirectionalStrategy:
    def __init__(self, config: BotConfig) -> None:
        self.config = config

    def _rsi(self, close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gains = delta.clip(lower=0).rolling(period).mean()
        losses = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gains / losses.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        close = data['close']
        returns = close.pct_change()
        fast_sma = close.rolling(self.config.fast_sma_period).mean()
        slow_sma = close.rolling(self.config.slow_sma_period).mean()
        rsi = self._rsi(close, self.config.rsi_period)
        volatility = returns.rolling(self.config.slow_sma_period).std()
        momentum = close.pct_change(self.config.momentum_lookback)

        row = pd.DataFrame(
            {
                'close': close,
                'fast': fast_sma,
                'slow': slow_sma,
                'rsi': rsi,
                'vol': volatility,
                'mom': momentum,
            }
        ).iloc[-1]

        if row.isna().any():
            return Signal('hold', 0.0, 0.0)

        trend_up = row['fast'] > row['slow'] and row['mom'] > 0
        trend_down = row['fast'] < row['slow'] and row['mom'] < 0
        vol_ok = self.config.min_volatility_pct <= row['vol'] <= self.config.max_volatility_pct
        long_filter = row['rsi'] < 65
        short_filter = row['rsi'] > 35

        if trend_up and vol_ok and long_filter:
            return Signal('buy', confidence=0.7, expected_move=float(abs(row['mom'])))
        if trend_down and vol_ok and short_filter:
            return Signal('sell', confidence=0.7, expected_move=float(abs(row['mom'])))
        return Signal('hold', 0.0, 0.0)


__all__ = ['DirectionalStrategy', 'Signal']
