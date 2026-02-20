from reco_trading.core.signal_fusion import SignalFusionEngine


def _seed_engine(engine: SignalFusionEngine) -> None:
    for i in range(60):
        engine.update_performance('momentum', 0.010 + (i % 5) * 0.001)
        engine.update_performance('mean_reversion', 0.001 + (i % 4) * 0.0002)
        engine.update_performance('volatility_breakout', 0.004 + (i % 6) * 0.0005)


def test_signal_fusion_returns_probability_between_zero_and_one() -> None:
    engine = SignalFusionEngine()
    _seed_engine(engine)

    probability = engine.fuse(
        signals={'momentum': 0.8, 'mean_reversion': 0.2, 'volatility_breakout': 0.1},
        regime='trend',
        volatility=0.3,
    )

    assert 0.0 <= probability <= 1.0


def test_dynamic_weights_favor_best_model() -> None:
    engine = SignalFusionEngine()
    _seed_engine(engine)

    weights = engine.compute_dynamic_weights()

    assert weights['momentum'] > weights['mean_reversion']


def test_regime_shift_changes_probability() -> None:
    engine = SignalFusionEngine()
    _seed_engine(engine)

    trend_prob = engine.fuse(
        signals={'momentum': 0.9, 'mean_reversion': -0.4},
        regime='trend',
        volatility=0.05,
        calibrate=False,
    )
    range_prob = engine.fuse(
        signals={'momentum': 0.9, 'mean_reversion': -0.4},
        regime='range',
        volatility=0.05,
        calibrate=False,
    )

    assert trend_prob != range_prob
