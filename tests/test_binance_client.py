import asyncio

import pytest

from reco_trading.infra import binance_client as bc


class _FakeExchange:
    def __init__(self):
        self.urls = {'api': {'public': '', 'private': ''}}
        self.loaded = False
        self.created = []

    def set_sandbox_mode(self, value: bool) -> None:
        self.sandbox = value

    async def load_markets(self):
        self.loaded = True

    async def create_market_buy_order(self, symbol, amount):
        self.created.append((symbol, amount))
        return {'id': '1', 'symbol': symbol, 'side': 'buy', 'amount': amount}

    async def close(self):
        return None


def test_binance_client_rejects_empty_credentials():
    with pytest.raises(ValueError, match='obligatorias'):
        bc.BinanceClient(api_key=' ', api_secret='x', testnet=True)


def test_create_market_order_initializes_markets(monkeypatch):
    fake = _FakeExchange()
    monkeypatch.setattr(bc.ccxt, 'binance', lambda *_a, **_k: fake)

    client = bc.BinanceClient(api_key='k', api_secret='s', testnet=True)
    out = asyncio.run(client.create_market_order('BTC/USDT', 'BUY', 0.01, firewall_checked=True))

    assert fake.loaded is True
    assert fake.created == [('BTC/USDT', 0.01)]
    assert out['id'] == '1'
    asyncio.run(client.close())


def test_ping_closes_exchange_when_initialize_fails(monkeypatch):
    class _FailingExchange(_FakeExchange):
        def __init__(self):
            super().__init__()
            self.closed = False

        async def load_markets(self):
            raise RuntimeError('boom')

        async def close(self):
            self.closed = True

    failing = _FailingExchange()
    monkeypatch.setattr(bc.ccxt, 'binance', lambda *_a, **_k: failing)

    client = bc.BinanceClient(api_key='k', api_secret='s', testnet=True)
    with pytest.raises(RuntimeError, match='Binance ping failed'):
        asyncio.run(client.ping())

    assert failing.closed is True
