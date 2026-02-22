from reco_trading.kernel.quant_kernel import DecisionEngine


def test_decision_engine_blocks_low_confidence_with_dynamic_threshold() -> None:
    engine = DecisionEngine(min_edge=0.0005)
    engine.update_context(
        momentum=0.61,
        reversion=0.59,
        global_probability=0.57,
        expected_edge=0.02,
        friction_cost=0.001,
        trading_enabled=True,
        market_operable=True,
        confidence_threshold=0.60,
        effective_min_edge=0.0004,
    )

    decision = engine.decide()

    assert decision == 'HOLD'
    assert 'confidence_below_threshold' in engine.last_reason


def test_decision_engine_blocks_uncertain_regime() -> None:
    engine = DecisionEngine(min_edge=0.0005)
    engine.update_context(
        momentum=0.80,
        reversion=0.20,
        global_probability=0.75,
        expected_edge=0.05,
        friction_cost=0.001,
        trading_enabled=True,
        market_operable=True,
        confidence_threshold=0.55,
        effective_min_edge=0.0003,
        regime_uncertain=True,
    )

    decision = engine.decide()

    assert decision == 'HOLD'
    assert 'regime_uncertain' in engine.last_reason
