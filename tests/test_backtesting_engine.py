from __future__ import annotations

import pandas as pd

from reco_trading.backtesting.engine import BacktestEngine


def _frame(rows: int = 220) -> pd.DataFrame:
    data = []
    price = 100.0
    for i in range(rows):
        drift = 0.2 if i % 15 < 7 else -0.15
        price = max(price + drift, 1.0)
        data.append([1_700_000_000_000 + i * 300_000, price - 1, price + 1, price - 2, price, 100 + (i % 10)])
    return pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])


def test_backtesting_engine_runs_and_produces_metrics() -> None:
    frame5 = _frame(220)
    frame15 = _frame(220)
    result = BacktestEngine(initial_equity=1000.0).run(frame5, frame15)

    assert isinstance(result.equity_curve, list)
    assert len(result.equity_curve) > 1
    assert result.metrics.max_drawdown >= 0
    assert 0 <= result.metrics.win_rate <= 1
