import asyncio

from reco_trading.core.execution_engine import ExecutionEngine
from reco_trading.core.microstructure import MicrostructureSnapshot


class _FakeClient:
    def __init__(self):
        self.last_amount = None

    async def fetch_balance(self):
        return {'USDT': {'free': 1000}, 'BTC': {'free': 2}}

    async def fetch_order_book(self, symbol, limit=10):
        return {'bids': [[100.0, 5.0], [99.9, 4.0]], 'asks': [[100.2, 5.0], [100.3, 4.0]]}

    async def create_market_order(self, symbol, side, amount):
        self.last_amount = amount
        return {'id': '1', 'symbol': symbol, 'side': side, 'amount': amount}

    async def wait_for_fill(self, symbol, order_id):
        return {'id': order_id, 'status': 'closed'}


class _SlowBalanceClient(_FakeClient):
    async def fetch_balance(self):
        await asyncio.sleep(0.05)
        return await super().fetch_balance()


class _FakeDB:
    def __init__(self):
        self.orders = []
        self.fills = []

    async def record_order(self, order):
        self.orders.append(order)

    async def record_fill(self, fill):
        self.fills.append(fill)


def test_execution_engine_validates_order_side_and_amount():
    engine = ExecutionEngine(_FakeClient(), 'BTC/USDT', _FakeDB(), redis_url='redis://localhost:6399/0')

    async def _run():
        assert await engine.execute_market_order('INVALID', 1) is None
        assert await engine.execute_market_order('BUY', -1) is None

    asyncio.run(_run())


def test_execution_engine_executes_market_order():
    db = _FakeDB()
    engine = ExecutionEngine(_FakeClient(), 'BTC/USDT', db, redis_url='redis://localhost:6399/0')

    async def _run():
        out = await engine.execute_market_order('BUY', 0.1)
        assert out is not None

    asyncio.run(_run())
    assert len(db.orders) == 1
    assert len(db.fills) == 1


def test_execution_engine_delegates_institutional_orders_to_sor():
    db = _FakeDB()
    engine = ExecutionEngine(
        _FakeClient(),
        'BTC/USDT',
        db,
        redis_url='redis://localhost:6399/0',
        institutional_order_threshold=1.0,
    )

    async def _run():
        out = await engine.execute('BUY', 2.0)
        assert out is not None
        assert out['status'] == 'institutional_completed'

    asyncio.run(_run())
    assert len(db.orders) > 1
    assert len(db.fills) > 1
