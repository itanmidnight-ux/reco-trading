from reco_trading.analysis.indicators import IndicatorSet
from reco_trading.analysis.regimes import MarketRegime
from reco_trading.core.market_state import MarketState
from reco_trading.kernel.decision_engine import DecisionEngine
from reco_trading.kernel.execution_engine import ExecutionGate
import pytest

from reco_trading.kernel.risk_manager import RiskManager


def _market_state(edge: float) -> MarketState:
    return MarketState(
        price=100.0,
        indicators=IndicatorSet(0.0, 0.2, 1.0, 0.0, 50.0, 22.0, 5.0, 0.7),
        regime=MarketRegime(name='TREND', volatility_extreme=False, tradable=True),
        expected_edge=edge,
        friction_cost=0.001,
        timestamp_ms=1,
    )


def test_decision_engine_applies_confidence_inertia() -> None:
    engine = DecisionEngine(inertia_old_weight=0.7)

    engine.update_context(_market_state(0.8))
    first = engine.decide()
    assert first.action == 'BUY'
    assert first.confidence == pytest.approx(0.27)
    assert first.scores['raw_confidence'] == 0.9

    engine.update_context(_market_state(0.8))
    second = engine.decide()
    assert second.confidence > first.confidence
    assert round(second.confidence, 4) == 0.459


def test_risk_manager_blocks_daily_loss_and_trade_frequency() -> None:
    manager = RiskManager(
        max_drawdown=0.2,
        max_latency_ms=400,
        max_spread_bps=40,
        max_risk_fraction=0.02,
        max_daily_loss=100.0,
        max_trades_per_hour=10,
    )

    blocked, reason = manager.evaluate_kill_switch(drawdown=0.01, latency_ms=10, spread_bps=5, daily_pnl=-120.0)
    assert blocked is True
    assert reason == 'daily_loss_limit'

    manager.kill_switch_active = False
    blocked, reason = manager.evaluate_kill_switch(drawdown=0.01, latency_ms=10, spread_bps=5, trades_last_hour=10)
    assert blocked is True
    assert reason == 'max_trades_per_hour'


def test_execution_gate_post_order_checks() -> None:
    gate = ExecutionGate(max_slippage_multiplier=2.0)

    partial = gate.validate_post_order(
        status='closed',
        requested_qty=1.0,
        executed_qty=0.8,
        expected_price=100.0,
        executed_price=100.1,
        expected_slippage=0.001,
    )
    assert partial.ok is False
    assert partial.reason == 'partial_fill'

    slipped = gate.validate_post_order(
        status='closed',
        requested_qty=1.0,
        executed_qty=1.0,
        expected_price=100.0,
        executed_price=100.5,
        expected_slippage=0.001,
    )
    assert slipped.ok is False
    assert slipped.reason == 'slippage_exceeded'

    rejected = gate.validate_post_order(
        status='expired',
        requested_qty=1.0,
        executed_qty=0.0,
        expected_price=100.0,
        executed_price=0.0,
        expected_slippage=0.001,
    )
    assert rejected.ok is False
    assert rejected.reason == 'order_expired'


def test_quant_kernel_requires_time_and_bars_for_warmup() -> None:
    pytest.importorskip('pydantic_settings')
    from reco_trading.kernel.quant_kernel import QuantKernel

    kernel = QuantKernel.__new__(QuantKernel)
    kernel.learning_started_at_ms = 0
    kernel.data_buffer = type('Buffer', (), {'ohlcv': [1] * 30})()
    kernel.MIN_WARMUP_SECONDS = 300
    kernel.MIN_WARMUP_BARS = 60

    ready, reason = QuantKernel._is_warmup_complete(kernel, now_ts=250.0)
    assert ready is False
    assert reason.startswith('warmup_active')

    kernel.data_buffer = type('Buffer', (), {'ohlcv': [1] * 61})()
    ready, reason = QuantKernel._is_warmup_complete(kernel, now_ts=301.0)
    assert ready is True
    assert reason == 'warmup_complete'


def test_quant_kernel_backfills_missing_runtime_state_fields() -> None:
    pytest.importorskip('pydantic_settings')
    from reco_trading.kernel.quant_kernel import QuantKernel

    kernel = QuantKernel.__new__(QuantKernel)
    kernel.state = type('LegacyState', (), {})()

    QuantKernel._ensure_runtime_state_fields(kernel)

    assert kernel.state.negative_edge_streak == 0
    assert kernel.state.binance_min_notional == 0.0
    assert kernel.state.final_order_notional == 0.0
