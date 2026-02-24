from __future__ import annotations

import numpy as np
import pandas as pd

from reco_trading.core.market_data import MarketDataService, MarketQuality
from reco_trading.kernel.conditional_performance import ConditionalPerformanceTracker
from reco_trading.kernel.edge_monitor import EdgeMonitor
from reco_trading.kernel.regime_controller import RegimeController
from reco_trading.kernel.risk_of_ruin import RiskOfRuinEstimator


def test_edge_monitor_outputs_valid_ranges() -> None:
    monitor = EdgeMonitor(window=50)
    for value in np.linspace(-0.001, 0.003, 80):
        out = monitor.update(float(value))
    assert 0.0 <= out.edge_confidence_score <= 1.0
    assert 0.0 <= out.p_value <= 1.0
    assert 0.0 <= out.bayesian_prob_edge_positive <= 1.0
    assert out.sprt_state in {'EDGE_HEALTHY_H0', 'EDGE_DECAY_H1', 'INCONCLUSIVE', 'INSUFFICIENT_DATA'}


def test_regime_controller_detects_liquidity_stress() -> None:
    controller = RegimeController(window=40)
    out = controller.update(volatility=0.02, autocorr=0.2, avg_spread_bps=60.0, relative_liquidity=0.3)
    assert out.current_regime == 'LIQUIDITY_STRESS'
    assert 0.0 <= out.regime_stability_score <= 1.0


def test_risk_of_ruin_is_conservative_when_edge_negative() -> None:
    estimator = RiskOfRuinEstimator()
    out = estimator.estimate(edge=-0.01, variance=0.0004, position_fraction=0.02, capital=1000.0)
    assert out.risk_of_ruin_probability >= 0.65
    assert 0.05 <= out.recommended_risk_multiplier <= 1.0


def test_conditional_performance_has_bounded_memory() -> None:
    tracker = ConditionalPerformanceTracker(window=30)
    for i in range(200):
        tracker.record_trade('HIGH_VOL_REGIME', pnl=float(i % 3 - 1), trade_return=float((i % 7 - 3) / 1000.0))
    summary = tracker.summary('HIGH_VOL_REGIME')
    assert summary['trades'] <= 30


def test_extended_modules_long_run_stability() -> None:
    monitor = EdgeMonitor(window=60)
    controller = RegimeController(window=60)
    tracker = ConditionalPerformanceTracker(window=60)
    ruin = RiskOfRuinEstimator()

    rng = np.random.default_rng(7)
    returns = rng.normal(0.0001, 0.002, size=1500)
    for r in returns:
        edge = monitor.update(float(r))
        regime = controller.update(
            volatility=float(abs(r) * 10.0),
            autocorr=float(np.clip(r * 100.0, -1.0, 1.0)),
            avg_spread_bps=float(8.0 + abs(r) * 1000.0),
            relative_liquidity=float(max(0.3, 1.2 - abs(r) * 100.0)),
        )
        tracker.record_trade(regime.current_regime, float(r * 1000.0), float(r))
        ruin_snapshot = ruin.estimate(edge=float(r), variance=1e-5 + float(abs(r)), position_fraction=0.01, capital=10000.0)

    assert edge.sprt_state in {'EDGE_HEALTHY_H0', 'EDGE_DECAY_H1', 'INCONCLUSIVE', 'INSUFFICIENT_DATA'}
    assert 0.0 <= ruin_snapshot.risk_of_ruin_probability <= 1.0
    assert tracker.summary(regime.current_regime)['trades'] <= 60


def test_market_quality_contract_with_quant_kernel_helper() -> None:
    from reco_trading.kernel.quant_kernel import QuantKernel

    kernel = QuantKernel.__new__(QuantKernel)

    class SettingsStub:
        market_min_avg_volume = 2.0

    kernel.s = SettingsStub()
    quality = MarketQuality(
        operable=True,
        reason='market_operable',
        spread_bps=10.0,
        realized_volatility=0.01,
        avg_volume=5.0,
        gap_ratio=0.0,
    )

    rel = QuantKernel._relative_liquidity_from_quality(kernel, quality)
    assert rel == 2.5


def test_market_quality_from_real_service_contract_with_kernel_helper() -> None:
    from reco_trading.kernel.quant_kernel import QuantKernel

    kernel = QuantKernel.__new__(QuantKernel)

    class SettingsStub:
        market_min_avg_volume = 10.0

    kernel.s = SettingsStub()
    service = MarketDataService(client=None, symbol='BTC/USDT', timeframe='5m')
    frame = pd.DataFrame(
        {
            'timestamp': pd.date_range('2024-01-01', periods=120, freq='5min', tz='UTC'),
            'open': np.linspace(100.0, 102.0, 120),
            'high': np.linspace(100.2, 102.2, 120),
            'low': np.linspace(99.8, 101.8, 120),
            'close': np.linspace(100.0, 102.0, 120),
            'volume': np.full(120, 25.0),
        }
    )

    quality = service.assess_market_quality(
        frame,
        spread_bps=5.0,
        max_spread_bps=20.0,
        max_volatility=1.0,
        min_avg_volume=1.0,
        max_gap_ratio=1.0,
    )

    rel = QuantKernel._relative_liquidity_from_quality(kernel, quality)
    assert rel == 2.5
