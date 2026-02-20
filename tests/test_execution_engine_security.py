import asyncio

from reco_trading.core.execution_engine import ExecutionEngine
from reco_trading.core.microstructure import MicrostructureSnapshot


class _FakeClient:
    def __init__(self):
        self.last_amount = None

    async def fetch_balance(self):
        return {'USDT': {'free': 1000}, 'BTC': {'free': 2}}

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


def test_execution_engine_applies_microstructure_adjustment():
    db = _FakeDB()
    client = _FakeClient()
    engine = ExecutionEngine(client, 'BTC/USDT', db, redis_url='redis://localhost:6399/0')

    micro = MicrostructureSnapshot(obi=0.0, cvd=0.0, spread=0.0, vpin=0.4, liquidity_shock=True)

    async def _run():
        out = await engine.execute_market_order('BUY', 1.0, microstructure=micro)
        assert out is not None

    asyncio.run(_run())
    assert client.last_amount is not None
    assert abs(client.last_amount - 0.21) < 1e-9


def test_execution_engine_rejects_invalid_microstructure_payload():
    engine = ExecutionEngine(_FakeClient(), 'BTC/USDT', _FakeDB(), redis_url='redis://localhost:6399/0')

    async def _run():
        assert await engine.execute_market_order('BUY', 1.0, microstructure='bad') is None

    asyncio.run(_run())


def test_execution_engine_uses_configurable_timeout_for_balance_check():
    db = _FakeDB()
    engine = ExecutionEngine(_SlowBalanceClient(), 'BTC/USDT', db, redis_url='redis://localhost:6399/0')

    async def _run():
        out = await engine.execute_market_order('BUY', 0.1, timeout_seconds=0.01, max_retries=1)
        assert out is None

    asyncio.run(_run())
    assert len(db.orders) == 0
    assert len(db.fills) == 0


def test_execute_keeps_backward_compatibility():
    db = _FakeDB()
    engine = ExecutionEngine(_FakeClient(), 'BTC/USDT', db, redis_url='redis://localhost:6399/0')

    async def _run():
        out = await engine.execute('BUY', 0.1)
        assert out is not None

    asyncio.run(_run())
    assert len(db.orders) == 1
    assert len(db.fills) == 1
