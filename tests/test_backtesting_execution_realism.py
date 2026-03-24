from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from reco_trading.backtesting.engine import BacktestEngine
from reco_trading.backtesting.simulator import TradeSimulator
from reco_trading.backtesting.validation import run_walk_forward_validation


def _frame(rows: int = 240) -> pd.DataFrame:
    data = []
    price = 100.0
    for i in range(rows):
        drift = 0.25 if i % 20 < 10 else -0.18
        price = max(price + drift, 1.0)
        data.append([1_700_000_000_000 + i * 300_000, price - 1, price + 1.8, price - 2.2, price, 600 + (i % 40) * 20])
    return pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])


def test_trade_simulator_tracks_fill_and_costs() -> None:
    sim = TradeSimulator()
    now = datetime.now(timezone.utc)
    opened = sim.open_position("BUY", quantity=2.0, price=100.0, timestamp=now, volatility_ratio=0.03, liquidity_ratio=0.4)
    assert opened is not None
    closed = sim.close_position(price=102.0, timestamp=now, volatility_ratio=0.02, liquidity_ratio=0.5)
    assert closed is not None
    assert closed.expected_fill_price is not None
    assert closed.realized_fill_price is not None
    assert closed.filled_quantity <= closed.quantity
    assert closed.commission_paid >= 0
    assert closed.spread_cost >= 0
    assert closed.slippage_cost >= 0


def test_backtest_engine_reports_net_cost_metrics() -> None:
    frame = _frame()
    result = BacktestEngine(initial_equity=1000.0).run(frame, frame)
    assert result.metrics.total_commissions >= 0
    assert result.metrics.total_spread_cost >= 0
    assert result.metrics.total_slippage_cost >= 0
    assert result.metrics.net_return_after_costs <= result.metrics.total_return


def test_walk_forward_validation_with_gap_and_benchmarks() -> None:
    frame = _frame(320)
    report = run_walk_forward_validation(frame, frame, train_window=120, test_window=60, gap_window=10)
    assert len(report.folds) >= 2
    for fold in report.folds:
        assert fold.test_start > fold.train_end
        assert isinstance(fold.buy_hold_return, float)
        assert isinstance(fold.ema_crossover_return, float)
    consolidated = report.consolidated
    assert set(consolidated) == {"strategy_return", "buy_hold_return", "ema_crossover_return"}
