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
    kernel._latest_edge_snapshot = SimpleNamespace(edge_confidence_score=0.5, t_stat=0.0, bayesian_prob_edge_positive=0.5, sprt_state='INCONCLUSIVE')
    kernel._latest_ruin_snapshot = SimpleNamespace(risk_of_ruin_probability=1.0)
    kernel._latest_regime_snapshot = {'regime_stability_score': 0.0}
    kernel._rolling_stats = {'expectancy': 0.0}
    kernel._signal_quality = {'volatility_adjusted_edge': 0.0, 'expected_value_net_costs': 0.0, 'stability_weight': 0.0}
    kernel.execution_status = 'IDLE'
    kernel.system_state = 'IDLE'
    kernel.s = SimpleNamespace(
        kill_switch_max_rejections=999,
        kill_switch_max_latency_ms=1_000_000.0,
        max_consecutive_losses=999,
        max_global_drawdown=1.0,
        max_daily_loss=1.0,
        allowed_sessions_utc=[],
    )
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


def test_check_kill_switch_state_is_pure_and_activate_records_trigger():
    kernel = _build_kernel_for_unit()
    kernel.s = SimpleNamespace(
        kill_switch_max_rejections=3,
        kill_switch_max_latency_ms=1_000.0,
        max_consecutive_losses=5,
        max_global_drawdown=0.99,
        max_daily_loss=0.99,
    )
    kernel.state.rejection_count = 3

    blocked, reason = kernel._check_kill_switch_state()

    assert blocked is True
    assert reason == 'too_many_rejections'
    assert kernel.state.kill_switch is False
    assert kernel.state.last_kill_switch_trigger is None

    blocked, reason = kernel._activate_kill_switch_if_needed()

    assert blocked is True
    assert reason == 'too_many_rejections'
    assert kernel.state.kill_switch is True
    assert kernel.state.last_kill_switch_trigger == 'too_many_rejections'


def test_publish_dashboard_does_not_activate_kill_switch() -> None:
    kernel = _build_kernel_for_unit()
    kernel.s = SimpleNamespace(
        kill_switch_max_rejections=1,
        kill_switch_max_latency_ms=1_000.0,
        max_consecutive_losses=5,
        max_global_drawdown=0.99,
        max_daily_loss=0.99,
        allowed_sessions_utc=[],
    )
    kernel._cooldown_seconds = lambda: 120.0
    kernel._is_within_allowed_session = lambda now: True
    kernel.conditional_performance = SimpleNamespace(summary=lambda regime: {'expectancy': 0.0})
    kernel._latest_edge_snapshot = SimpleNamespace(
        edge_confidence_score=0.5,
        t_stat=0.0,
        bayesian_prob_edge_positive=0.5,
        sprt_state='INCONCLUSIVE',
    )
    kernel._latest_ruin_snapshot = SimpleNamespace(risk_of_ruin_probability=1.0)
    kernel._latest_regime_snapshot = {'regime_stability_score': 0.0}
    kernel._rolling_stats = {'expectancy': 0.0}
    kernel._signal_quality = {'volatility_adjusted_edge': 0.0, 'expected_value_net_costs': 0.0, 'stability_weight': 0.0}
    kernel.decision_engine = SimpleNamespace(last_confidence=0.0, last_scores={}, last_reason='none')
    kernel.execution_status = 'IDLE'
    kernel.system_state = 'IDLE'
    kernel.state.rejection_count = 1

    kernel._publish_dashboard('HOLD', 0.0, 0.5, 0.5, 0.5, 'RANGE', 100.0, 'OK')

    assert kernel.state.kill_switch is False
    assert kernel.state.last_kill_switch_trigger is None
    assert kernel.dashboard.updates


def test_publish_dashboard_uses_exchange_equity_as_capital_actual() -> None:
    kernel = _build_kernel_for_unit()
    kernel.s = SimpleNamespace(
        kill_switch_max_rejections=1,
        kill_switch_max_latency_ms=1_000.0,
        max_consecutive_losses=5,
        max_global_drawdown=0.99,
        max_daily_loss=0.99,
        allowed_sessions_utc=[],
    )
    kernel._cooldown_seconds = lambda: 120.0
    kernel._is_within_allowed_session = lambda now: True
    kernel.conditional_performance = SimpleNamespace(summary=lambda regime: {'expectancy': 0.0})
    kernel._latest_edge_snapshot = SimpleNamespace(
        edge_confidence_score=0.5,
        t_stat=0.0,
        bayesian_prob_edge_positive=0.5,
        sprt_state='INCONCLUSIVE',
    )
    kernel._latest_ruin_snapshot = SimpleNamespace(risk_of_ruin_probability=1.0)
    kernel._latest_regime_snapshot = {'regime_stability_score': 0.0}
    kernel._rolling_stats = {'expectancy': 0.0}
    kernel._signal_quality = {'volatility_adjusted_edge': 0.0, 'expected_value_net_costs': 0.0, 'stability_weight': 0.0}
    kernel.decision_engine = SimpleNamespace(last_confidence=0.0, last_scores={}, last_reason='none')
    kernel.execution_status = 'IDLE'
    kernel.system_state = 'IDLE'
    kernel.state.exchange_equity = 470.49
    kernel.state.unrealized_pnl = 12.34
    kernel._daily_anchor_equity = 470.49
    kernel._daily_anchor_date = __import__('datetime').datetime.now(__import__('datetime').timezone.utc).date()

    kernel._publish_dashboard('HOLD', 0.0, 0.5, 0.5, 0.5, 'RANGE', 100.0, 'OK')

    assert kernel.dashboard.updates
    snapshot = kernel.dashboard.updates[-1]
    assert snapshot.equity == 470.49
    assert snapshot.pnl == 12.34
