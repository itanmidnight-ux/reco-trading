from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class BacktestResult:
    cumulative_return: float
    sharpe: float
    hit_rate: float
    observations: int


class AlphaFactor(ABC):
    def __init__(self, name: str, lookback: int = 20) -> None:
        self.name = name
        self.lookback = int(lookback)

    @abstractmethod
    def compute(self, frame: pd.DataFrame) -> pd.Series:
        """Compute raw factor signal."""

    def validate(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            raise ValueError(f'{self.name}: input frame is empty')
        if len(frame) < self.lookback:
            raise ValueError(f'{self.name}: expected at least {self.lookback} rows, got {len(frame)}')
        if 'close' not in frame.columns:
            raise ValueError(f'{self.name}: required column "close" is missing')

    def backtest(self, factor_values: pd.Series, future_returns: pd.Series, cost_bps: float = 0.0) -> BacktestResult:
        aligned = pd.concat([factor_values, future_returns], axis=1).dropna()
        aligned.columns = ['factor', 'future_return']
        if aligned.empty:
            return BacktestResult(0.0, 0.0, 0.0, 0)

        signal = np.sign(aligned['factor']).astype(float)
        gross = signal * aligned['future_return']
        turnover = signal.diff().abs().fillna(0.0)
        net = gross - turnover * (cost_bps / 10_000)

        sharpe_den = net.std(ddof=0)
        sharpe = float((net.mean() / sharpe_den) * np.sqrt(252)) if sharpe_den > 0 else 0.0
        hit_rate = float((net > 0).mean())
        cumulative_return = float((1 + net).prod() - 1)

        return BacktestResult(
            cumulative_return=cumulative_return,
            sharpe=sharpe,
            hit_rate=hit_rate,
            observations=int(len(aligned)),
        )
