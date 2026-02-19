from __future__ import annotations

import numpy as np
import pandas as pd

from reco_trading.research.metrics import sharpe, sortino, max_drawdown, calmar, profit_factor


class BacktestEngine:
    def __init__(self, fee_rate: float, slippage_bps: float) -> None:
        self.fee_rate = fee_rate
        self.slippage = slippage_bps / 10_000

    def run(self, frame: pd.DataFrame, signals: pd.Series) -> dict:
        ret = frame['return'].shift(-1).fillna(0)
        strat = ret * signals.replace({'BUY': 1, 'SELL': -1, 'HOLD': 0})
        costs = (signals != 'HOLD').astype(float) * (self.fee_rate + self.slippage)
        net = strat - costs
        equity = (1 + net).cumprod()
        returns = net.to_numpy()
        return {
            'sharpe': sharpe(returns),
            'sortino': sortino(returns),
            'max_drawdown': max_drawdown(equity),
            'calmar': calmar(returns, equity),
            'profit_factor': profit_factor(returns),
        }
