import asyncio
from dataclasses import dataclass

from reco_trading.core.execution_engine import ExecutionEngine
from reco_trading.core.microstructure import MicrostructureSnapshot


class _FakeClient:
    def __init__(self):
        self.last_amount = None

    async def fetch_balance(self):
        return {'USDT': {'free': 1000}, 'BTC': {'free': 2}}

    async def fetch_order_book(self, symbol, limit=10):
        return {'bids': [[100.0, 5.0], [99.9, 4.0]], 'asks': [[100.2, 5.0], [100.3, 4.0]]}

    async def create_market_order(self, symbol, side, amount, **kwargs):
        self.last_amount = amount
        return {'id': '1', 'symbol': symbol, 'side': side, 'amount': amount}

    async def wait_for_fill(self, symbol, order_id):
        return {'id': order_id, 'status': 'closed'}


class _CreateTimeoutClient(_FakeClient):
    async def create_market_order(self, symbol, side, amount, **kwargs):
        await asyncio.sleep(0.2)
        return await super().create_market_order(symbol, side, amount)


class _FillTimeoutClient(_FakeClient):
    async def wait_for_fill(self, symbol, order_id):
        await asyncio.sleep(0.2)
        return await super().wait_for_fill(symbol, order_id)


class _FakeDB:
    def __init__(self):
        self.orders = []
        self.fills = []
        self.executions = []

    async def record_order(self, order):
        self.orders.append(order)

    async def record_fill(self, fill):
        self.fills.append(fill)

    async def persist_order_execution(self, execution):
        self.executions.append(execution)


@dataclass
class _Decision:
    allowed: bool
    reason: str
    risk_snapshot: dict
    recommended_size: float


class _AllowFirewall:
    def __init__(self):
        self.registered = 0

    async def evaluate(self, **kwargs):
        return _Decision(True, 'allowed', {'ok': True}, kwargs['amount'])

    def register_fill(self, **kwargs):
        self.registered += 1


class _RejectFirewall:
    async def evaluate(self, **kwargs):
        return _Decision(False, 'daily_notional_limit', {'amount': kwargs['amount']}, 0.0)

    def register_fill(self, **kwargs):
        raise AssertionError('register_fill no debería ejecutarse cuando el firewall rechaza')


class _FakeQuantKernel:
    def __init__(self):
        self.rejections = []
        self.blocked = False

    def should_block_trading(self):
        return self.blocked

    def on_firewall_rejection(self, reason, risk_snapshot):
        self.rejections.append((reason, risk_snapshot))


def test_execution_engine_validates_order_side_and_amount():
    engine = ExecutionEngine(_FakeClient(), 'BTC/USDT', _FakeDB(), redis_url='redis://localhost:6399/0')

    async def _run():
        assert await engine.execute_market_order('INVALID', 1) is None
        assert await engine.execute_market_order('BUY', -1) is None

    asyncio.run(_run())


def test_execution_engine_executes_market_order_without_microstructure():
    db = _FakeDB()
    firewall = _AllowFirewall()
    engine = ExecutionEngine(_FakeClient(), 'BTC/USDT', db, redis_url='redis://localhost:6399/0', firewall=firewall)

    async def _run():
        out = await engine.execute_market_order('BUY', 0.1)
        assert out is not None

    asyncio.run(_run())
    assert len(db.orders) == 1
    assert len(db.fills) == 1
    assert len(db.executions) == 1
    assert firewall.registered == 1


def test_execution_engine_executes_market_order_with_microstructure():
    db = _FakeDB()
    engine = ExecutionEngine(_FakeClient(), 'BTC/USDT', db, redis_url='redis://localhost:6399/0')
    microstructure = MicrostructureSnapshot(obi=0.1, cvd=1.0, spread=0.0005, vpin=0.4, liquidity_shock=False)

    async def _run():
        out = await engine.execute_market_order('BUY', 0.1, microstructure=microstructure)
        assert out is not None

    asyncio.run(_run())
    assert len(db.orders) == 1
    assert len(db.fills) == 1
    assert len(db.executions) == 1


def test_execution_engine_times_out_during_create_market_order():
    db = _FakeDB()
    engine = ExecutionEngine(_CreateTimeoutClient(), 'BTC/USDT', db, redis_url='redis://localhost:6399/0')

    async def _run():
        out = await engine.execute_market_order('BUY', 0.1, timeout_seconds=0.1, max_retries=1)
        assert out is None

    asyncio.run(_run())
    assert len(db.orders) == 0
    assert len(db.fills) == 0
    assert len(db.executions) == 0


def test_execution_engine_times_out_waiting_for_fill():
    db = _FakeDB()
    engine = ExecutionEngine(_FillTimeoutClient(), 'BTC/USDT', db, redis_url='redis://localhost:6399/0')

    async def _run():
        out = await engine.execute_market_order('BUY', 0.1, timeout_seconds=0.1, max_retries=1)
        assert out is None

    asyncio.run(_run())
    assert len(db.orders) == 1
    assert len(db.fills) == 0


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


def test_execution_engine_blocks_order_when_firewall_rejects():
    db = _FakeDB()
    kernel = _FakeQuantKernel()
    engine = ExecutionEngine(
        _FakeClient(),
        'BTC/USDT',
        db,
        redis_url='redis://localhost:6399/0',
        firewall=_RejectFirewall(),
        quant_kernel=kernel,
    )

    async def _run():
        out = await engine.execute_market_order('BUY', 0.5)
        assert out is None

    asyncio.run(_run())
    assert kernel.rejections
    assert len(db.orders) == 0


def test_execution_engine_blocks_when_quant_kernel_kill_switch_active():
    db = _FakeDB()
    kernel = _FakeQuantKernel()
    kernel.blocked = True
    engine = ExecutionEngine(
        _FakeClient(),
        'BTC/USDT',
        db,
        redis_url='redis://localhost:6399/0',
        firewall=_AllowFirewall(),
        quant_kernel=kernel,
    )

    async def _run():
        out = await engine.execute('BUY', 0.1)
        assert out is None

    asyncio.run(_run())
    assert len(db.orders) == 0


def test_execution_engine_accepts_lowercase_side_input():
    db = _FakeDB()
    firewall = _AllowFirewall()
    engine = ExecutionEngine(_FakeClient(), 'BTC/USDT', db, redis_url='redis://localhost:6399/0', firewall=firewall)

    async def _run():
        out = await engine.execute_market_order('buy', 0.1)
        assert out is not None

    asyncio.run(_run())
    assert len(db.orders) == 1
    assert firewall.registered == 1
