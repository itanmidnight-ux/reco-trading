import math

import pandas as pd

from reco_trading.research.backtest_engine import BacktestEngine
from reco_trading.research.metrics import (
    adverse_selection_cost,
    aggregate_execution_quality,
    microstructure_alpha_decay,
    normalize_fills,
    quote_fill_ratio,
    realized_spread,
    unrealized_spread,
)


def test_metrics_edge_cases_without_fills_and_zero_spread():
    quotes = [
        {'timestamp': '2025-01-01T00:00:00Z', 'bid': 100.0, 'ask': 100.0, 'bid_size': 5.0, 'ask_size': 5.0},
        {'timestamp': '2025-01-01T00:01:00Z', 'bid': 100.0, 'ask': 100.0, 'bid_size': 2.0, 'ask_size': 2.0},
    ]
    mids = [
        {'timestamp': '2025-01-01T00:00:00Z', 'midprice': 100.0},
        {'timestamp': '2025-01-01T00:01:00Z', 'midprice': 100.0},
    ]

    assert quote_fill_ratio([], quotes) == 0.0
    assert adverse_selection_cost([], mids) == 0.0
    assert realized_spread([], mids) == 0.0
    assert unrealized_spread([], mids) == 0.0
    assert microstructure_alpha_decay([], mids) == {'h1': 0.0, 'h5': 0.0, 'h10': 0.0}


def test_normalization_sorts_unordered_timestamps():
    fills = [
        {'timestamp': '2025-01-01T00:03:00Z', 'side': 'BUY', 'price': 101.0, 'quantity': 1.0},
        {'timestamp': '2025-01-01T00:01:00Z', 'side': 'SELL', 'price': 102.0, 'quantity': 1.0},
        {'timestamp': '2025-01-01T00:02:00Z', 'side': 'BUY', 'price': 100.0, 'quantity': 1.0},
    ]

    normalized = normalize_fills(fills)
    assert [f.timestamp.isoformat() for f in normalized] == [
        '2025-01-01T00:01:00+00:00',
        '2025-01-01T00:02:00+00:00',
        '2025-01-01T00:03:00+00:00',
    ]


def test_aggregate_execution_quality_by_exchange_and_strategy():
    fills = [
        {'timestamp': '2025-01-01T00:00:00Z', 'side': 'BUY', 'price': 100.0, 'quantity': 1.0, 'exchange': 'BINANCE', 'strategy': 'mm'},
        {'timestamp': '2025-01-01T00:01:00Z', 'side': 'SELL', 'price': 101.0, 'quantity': 1.0, 'exchange': 'BYBIT', 'strategy': 'arb'},
    ]
    quotes = [
        {'timestamp': '2025-01-01T00:00:00Z', 'bid': 99.9, 'ask': 100.1, 'bid_size': 2.0, 'ask_size': 2.0, 'exchange': 'BINANCE', 'strategy': 'mm'},
        {'timestamp': '2025-01-01T00:01:00Z', 'bid': 100.9, 'ask': 101.1, 'bid_size': 3.0, 'ask_size': 3.0, 'exchange': 'BYBIT', 'strategy': 'arb'},
    ]
    mids = [
        {'timestamp': '2025-01-01T00:00:00Z', 'midprice': 100.0, 'exchange': 'BINANCE', 'strategy': 'mm'},
        {'timestamp': '2025-01-01T00:01:00Z', 'midprice': 101.0, 'exchange': 'BYBIT', 'strategy': 'arb'},
    ]

    result = aggregate_execution_quality(fills, quotes, mids)

    assert set(result['by_exchange']) == {'BINANCE', 'BYBIT'}
    assert set(result['by_strategy']) == {'mm', 'arb'}
    assert math.isfinite(result['global']['inventory_turnover'])


def test_backtest_engine_includes_execution_quality_payload():
    n = 20
    frame = pd.DataFrame(
        {
            'close': [100.0 + i * 0.1 for i in range(n)],
            'high': [100.2 + i * 0.1 for i in range(n)],
            'low': [99.8 + i * 0.1 for i in range(n)],
            'return': [0.001 if i % 2 == 0 else -0.0005 for i in range(n)],
            'exchange': ['SIMX' for _ in range(n)],
            'strategy': ['test_strat' for _ in range(n)],
        }
    )
    signals = pd.Series(['BUY' if i % 3 == 0 else 'HOLD' for i in range(n)])

    stats = BacktestEngine(fee_rate=0.0004, slippage_bps=3).run(frame=frame, signals=signals)

    assert 'execution_quality' in stats
    assert 'global' in stats['execution_quality']
    assert 'SIMX' in stats['execution_quality']['by_exchange']
    assert 'test_strat' in stats['execution_quality']['by_strategy']
