from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import asyncio

import pytest

from reco_trading.hft.multi_exchange_arbitrage import (
    ExchangeAdapter,
    ExchangeAdapterFactory,
    MultiExchangeArbitrageEngine,
)


@dataclass
class StubAdapter(ExchangeAdapter):
    name: str
    order_book: dict[str, Any]

    async def get_order_book(self, symbol: str, limit: int = 20) -> dict[str, Any]:
        return self.order_book

    async def get_ticker(self, symbol: str) -> dict[str, Any]:
        return {'symbol': symbol}

    async def create_order(self, symbol: str, side: str, amount: float, order_type: str = 'market') -> dict[str, Any]:
        return {'id': f'{self.name}-{side}', 'symbol': symbol, 'amount': amount, 'type': order_type}

    async def get_balance(self) -> dict[str, Any]:
        return {'USDT': {'free': 1000}}

    async def close(self) -> None:
        return None


class DummyRegisteredAdapter(ExchangeAdapter):
    def __init__(self, config: dict[str, Any]) -> None:
        self.name = config.get('name', 'dummy')

    async def get_order_book(self, symbol: str, limit: int = 20) -> dict[str, Any]:
        return {'bids': [[100, 1]], 'asks': [[101, 1]]}

    async def get_ticker(self, symbol: str) -> dict[str, Any]:
        return {'symbol': symbol}

    async def create_order(self, symbol: str, side: str, amount: float, order_type: str = 'market') -> dict[str, Any]:
        return {'id': '1'}

    async def get_balance(self) -> dict[str, Any]:
        return {}

    async def close(self) -> None:
        return None


def test_scan_symbol_uses_expected_spread_formula() -> None:
    engine = MultiExchangeArbitrageEngine(
        adapters={
            'a': StubAdapter(name='a', order_book={'bids': [[102.0, 1.0]], 'asks': [[103.0, 1.0]]}),
            'b': StubAdapter(name='b', order_book={'bids': [[99.0, 1.0]], 'asks': [[100.0, 1.0]]}),
        },
        min_edge_bps=1,
    )

    opportunities = asyncio.run(engine.scan_symbol('BTC/USDT'))

    assert len(opportunities) == 1
    best = opportunities[0]
    expected_mid = (102.0 + 100.0) / 2
    expected_spread = (102.0 - 100.0) / expected_mid
    assert best.spread == pytest.approx(expected_spread)
    assert best.expected_edge_bps == pytest.approx(expected_spread * 10_000)
    assert best.sell_exchange == 'a'
    assert best.buy_exchange == 'b'


def test_execute_opportunity_returns_report() -> None:
    engine = MultiExchangeArbitrageEngine(
        adapters={
            'a': StubAdapter(name='a', order_book={'bids': [[101.0, 1.0]], 'asks': [[102.0, 1.0]]}),
            'b': StubAdapter(name='b', order_book={'bids': [[99.0, 1.0]], 'asks': [[100.0, 1.0]]}),
        },
        min_edge_bps=1,
    )

    opportunity = asyncio.run(engine.scan_symbol('ETH/USDT'))[0]
    report = asyncio.run(engine.execute_opportunity(opportunity, amount=0.5))

    assert report.status == 'submitted'
    assert report.buy_order_id == 'b-buy'
    assert report.sell_order_id == 'a-sell'


def test_exchange_factory_registry_is_extensible() -> None:
    ExchangeAdapterFactory.register('dummy', DummyRegisteredAdapter)

    adapters = ExchangeAdapterFactory.create_from_config(
        {
            'dummy': {'enabled': True, 'name': 'dummy-exchange'},
            'binance': {'enabled': False},
        }
    )

    assert 'dummy' in adapters
    assert adapters['dummy'].name == 'dummy-exchange'
