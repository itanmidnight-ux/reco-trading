import asyncio
from types import SimpleNamespace

from reco_trading.kernel.quant_kernel import QuantKernel, RuntimeState


class _DummyMonitoring:
    def __init__(self):
        self.calls = 0

    def set_system_degraded(self, error):
        self.calls += 1


class _DummyDashboard:
    def __init__(self):
        self.updates = []

    def update(self, snapshot):
        self.updates.append(snapshot)


def _build_kernel_for_unit() -> QuantKernel:
    kernel = QuantKernel.__new__(QuantKernel)
    kernel.MAX_CONSECUTIVE_CYCLE_ERRORS = 5
    kernel.state = RuntimeState()
    kernel.monitoring = _DummyMonitoring()
    kernel.dashboard = _DummyDashboard()
    kernel.shutdown_event = asyncio.Event()
    kernel._shutdown_reason = 'running'
    kernel.initial_equity = 10_000.0
    kernel.risk_manager = SimpleNamespace(config=SimpleNamespace(risk_per_trade=0.005))
    return kernel


def test_cycle_exception_does_not_shutdown_before_threshold():
    kernel = _build_kernel_for_unit()

    should_shutdown = kernel._handle_cycle_exception(ValueError('boom'))

    assert should_shutdown is False
    assert kernel.state.consecutive_cycle_errors == 1
    assert kernel.monitoring.calls == 1
    assert not kernel.shutdown_event.is_set()
    assert kernel.dashboard.updates


def test_cycle_exception_triggers_shutdown_at_threshold():
    kernel = _build_kernel_for_unit()
    kernel.state.consecutive_cycle_errors = 4

    should_shutdown = kernel._handle_cycle_exception(RuntimeError('fatal'))

    assert should_shutdown is True
    assert kernel.state.consecutive_cycle_errors == 5
    assert kernel.shutdown_event.is_set()
    assert kernel._shutdown_reason == 'max_consecutive_cycle_errors'


class _DummyExecutionEngine:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def execute(self, side, amount):
        self.calls.append({'side': side, 'amount': amount})
        return self.response


def test_execute_order_returns_none_for_invalid_qty():
    kernel = _build_kernel_for_unit()
    kernel.execution_engine = _DummyExecutionEngine(response={'status': 'closed'})
    kernel.s = SimpleNamespace(symbol='BTC/USDT')

    result = asyncio.run(kernel._execute_order('BUY', 0.0))

    assert result is None
    assert kernel.execution_engine.calls == []


def test_execute_order_normalizes_single_fill_payload():
    kernel = _build_kernel_for_unit()
    kernel.execution_engine = _DummyExecutionEngine(
        response={
            'id': 'ord-1',
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'filled': 0.01,
            'average': 101_000,
            'status': 'closed',
        }
    )
    kernel.s = SimpleNamespace(symbol='BTC/USDT')

    result = asyncio.run(kernel._execute_order('BUY', 0.01))

    assert result is not None
    assert result['qty'] == 0.01
    assert result['price'] == 101_000
    assert result['status'] == 'closed'
    assert result['side'] == 'BUY'
    assert result['symbol'] == 'BTC/USDT'
    assert result['pnl'] == 0.0


def test_execute_order_returns_none_when_execution_unfilled():
    kernel = _build_kernel_for_unit()
    kernel.execution_engine = _DummyExecutionEngine(response={'status': 'open', 'filled': 0.0})
    kernel.s = SimpleNamespace(symbol='BTC/USDT')

    result = asyncio.run(kernel._execute_order('SELL', 0.1))

    assert result is None
