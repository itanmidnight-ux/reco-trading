import asyncio

import pytest

from reco_trading.infra import binance_client as bc


class _FakeExchange:
    def __init__(self):
        self.urls = {'api': {'public': '', 'private': ''}}
        self.loaded = False
        self.created = []
        self.created_orders = []

    def set_sandbox_mode(self, value: bool) -> None:
        self.sandbox = value

    async def load_markets(self):
        self.loaded = True


    async def create_order(self, symbol, order_type, side, amount, price=None, params=None):
        self.created_orders.append(
            {
                'symbol': symbol,
                'type': order_type,
                'side': side,
                'amount': amount,
                'price': price,
                'params': params or {},
            }
        )
        payload = {'id': '2', 'symbol': symbol, 'side': side, 'amount': amount, 'type': order_type}
        payload.update(params or {})
        return payload

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


def test_create_market_order_with_client_id_uses_create_order_payload(monkeypatch):
    fake = _FakeExchange()
    monkeypatch.setattr(bc.ccxt, 'binance', lambda *_a, **_k: fake)

    client = bc.BinanceClient(api_key='k', api_secret='s', testnet=True)
    out = asyncio.run(
        client.create_market_order_with_client_id(
            'BTC/USDT',
            'BUY',
            0.01,
            client_order_id='abc-123',
            firewall_checked=True,
        )
    )

    assert fake.created_orders
    order = fake.created_orders[-1]
    assert order['symbol'] == 'BTC/USDT'
    assert order['type'] == 'market'
    assert order['side'] == 'buy'
    assert order['amount'] == 0.01
    assert order['params'].get('newClientOrderId') == 'abc-123'
    assert out.get('newClientOrderId') == 'abc-123'
    asyncio.run(client.close())
