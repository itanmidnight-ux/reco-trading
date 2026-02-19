from __future__ import annotations

import numpy as np
import pandas as pd

from reco_trading.research.metrics import calmar, max_drawdown, profit_factor, sharpe, sortino


class BacktestEngine:
    def __init__(self, fee_rate: float, slippage_bps: float) -> None:
        self.fee_rate = fee_rate
        self.slippage = slippage_bps / 10_000

    def run(self, frame: pd.DataFrame, signals: pd.Series) -> dict:
        ret_next = frame['return'].shift(-1).fillna(0)
        side = signals.map({'BUY': 1, 'SELL': -1, 'HOLD': 0}).fillna(0)
        gross = ret_next * side

        trades = side.diff().fillna(0).abs().clip(upper=1)
        costs = trades * (self.fee_rate + self.slippage)
        net = gross - costs

        equity = np.cumprod(1 + net.to_numpy())
        returns = net.to_numpy()

        return {
            'sharpe': sharpe(returns),
            'sortino': sortino(returns),
            'max_drawdown': max_drawdown(equity),
            'calmar': calmar(returns, equity),
            'profit_factor': profit_factor(returns),
            'trades': int(trades.sum()),
        }
