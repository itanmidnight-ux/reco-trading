from __future__ import annotations

import numpy as np
import pandas as pd

from reco_trading.research.metrics import calmar, expectancy, max_drawdown, profit_factor, sharpe, sortino, ulcer_index


class BacktestEngine:
    def __init__(self, fee_rate: float, slippage_bps: float, latency_bars: int = 1) -> None:
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps / 10_000
        self.latency_bars = max(latency_bars, 0)

    def run(
        self,
        frame: pd.DataFrame,
        signals: pd.Series,
        partial_fill_ratio: float = 0.9,
        latency_slippage_inflation: float | pd.Series | None = None,
        execution_probability: float | pd.Series | None = None,
    ) -> dict:
        side = signals.map({"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}).fillna(0.0)
        side_exec = side.shift(self.latency_bars).fillna(0.0) * float(np.clip(partial_fill_ratio, 0.1, 1.0))

        exec_prob = execution_probability
        if exec_prob is None:
            exec_prob_series = pd.Series(1.0, index=frame.index)
        elif np.isscalar(exec_prob):
            exec_prob_series = pd.Series(float(exec_prob), index=frame.index)
        else:
            exec_prob_series = pd.Series(exec_prob, index=frame.index)
        exec_prob_series = exec_prob_series.fillna(1.0).clip(lower=0.0, upper=1.0)
        side_exec = side_exec * exec_prob_series

        returns = frame["return"].shift(-1).fillna(0.0)
        spread = (frame["high"] - frame["low"]) / frame["close"].replace(0, np.nan)
        dyn_slippage = self.slippage_bps * (1.0 + spread.fillna(spread.median()).clip(lower=0.0))

        inflation = latency_slippage_inflation
        if inflation is None:
            inflation_series = pd.Series(1.0, index=frame.index)
        elif np.isscalar(inflation):
            inflation_series = pd.Series(float(inflation), index=frame.index)
        else:
            inflation_series = pd.Series(inflation, index=frame.index)
        inflation_series = inflation_series.fillna(1.0).clip(lower=1.0)
        dyn_slippage = dyn_slippage * inflation_series

        gross = side_exec * returns
        trades = side_exec.diff().abs().clip(upper=1.0)
        costs = trades * (self.fee_rate + dyn_slippage)
        net = (gross - costs).fillna(0.0)

        eq = np.cumprod(1.0 + net.to_numpy(dtype=float))
        ret = net.to_numpy(dtype=float)

        return {
            "expectancy": expectancy(ret),
            "sharpe": sharpe(ret),
            "sortino": sortino(ret),
            "calmar": calmar(ret, eq),
            "max_drawdown": max_drawdown(eq),
            "ulcer_index": ulcer_index(eq),
            "profit_factor": profit_factor(ret),
            "trades": int(trades.sum()),
            "terminal_equity": float(eq[-1]) if len(eq) else 1.0,
        }
