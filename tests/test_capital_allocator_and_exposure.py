from __future__ import annotations

import pandas as pd

from reco_trading.core.portfolio_exposure_manager import PortfolioExposureManager
from reco_trading.hft.capital_allocator import AllocationLimits, AllocationRequest, CapitalAllocator


def test_portfolio_exposure_groups_by_exchange() -> None:
    manager = PortfolioExposureManager(lookback=3)
    returns = pd.DataFrame({'BTC/USDT': [0.01, -0.01, 0.005], 'ETH/USDT': [0.02, -0.005, 0.01]})
    notionals = {
        'binance:BTC/USDT': 1000.0,
        'kraken:BTC/USDT': -500.0,
        'binance:ETH/USDT': 300.0,
    }

    snapshot = manager.evaluate(returns, notionals)

    assert snapshot.exchange_notionals['binance'] == 1300.0
    assert snapshot.exchange_notionals['kraken'] == -500.0
    assert snapshot.cross_exchange_notional == 1800.0


def test_capital_allocator_applies_exchange_and_cross_limits() -> None:
    allocator = CapitalAllocator(
        AllocationLimits(
            max_global_notional=10_000,
            max_per_exchange_notional={'binance': 4_000, 'kraken': 4_000},
            max_single_opportunity_notional=2_500,
        ),
        capital_isolation={'binance': 0.3, 'kraken': 0.3},
        max_cross_exchange_notional=8_000,
    )

    result = allocator.allocate(
        AllocationRequest(
            symbol='BTC/USDT',
            buy_exchange='binance',
            sell_exchange='kraken',
            mid_price=100.0,
            expected_edge_bps=12,
            correlation_score=0.2,
        ),
        equity=10_000,
        inventory_by_exchange={'binance': 0.1, 'kraken': 0.05},
        notionals_by_exchange={'binance': 2_000, 'kraken': 1_500},
        notionals_by_asset={'BTC/USDT': 1_000},
    )

    assert result.allowed is True
    assert result.notional <= 1_000
    assert result.units > 0
