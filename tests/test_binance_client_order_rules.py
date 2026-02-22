import asyncio

import pytest

from reco_trading.infra.binance_client import BinanceClient


class _FakeExchange:
    def __init__(self):
        self.market_calls = 0

    def amount_to_precision(self, symbol, amount):
        return f"{amount:.4f}"

    def market(self, symbol):
        self.market_calls += 1
        return {
            'limits': {
                'amount': {'min': 0.001, 'max': 1.0},
                'cost': {'min': 10.0},
            }
        }


@pytest.mark.parametrize('amount,price', [(0.01, 2000.0), (0.5, 100.0)])
def test_sanitize_order_quantity_accepts_valid_amount(amount, price):
    client = BinanceClient('k', 's', testnet=True)
    client.exchange = _FakeExchange()

    async def _run():
        client.initialize = lambda: asyncio.sleep(0)
        out = await client.sanitize_order_quantity('BTC/USDT', amount=amount, reference_price=price)
        assert out > 0

    asyncio.run(_run())


def test_sanitize_order_quantity_uses_cached_rules():
    client = BinanceClient('k', 's', testnet=True)
    fake = _FakeExchange()
    client.exchange = fake

    async def _run():
        client.initialize = lambda: asyncio.sleep(0)
        await client.get_symbol_rules('BTC/USDT')
        await client.get_symbol_rules('BTC/USDT')

    asyncio.run(_run())
    assert fake.market_calls == 1


def test_sanitize_order_quantity_rejects_notional_below_minimum():
    client = BinanceClient('k', 's', testnet=True)
    client.exchange = _FakeExchange()

    async def _run():
        client.initialize = lambda: asyncio.sleep(0)
        with pytest.raises(ValueError, match='Notional'):
            await client.sanitize_order_quantity('BTC/USDT', amount=0.001, reference_price=100.0)

    asyncio.run(_run())
