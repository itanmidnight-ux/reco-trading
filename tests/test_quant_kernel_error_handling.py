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
