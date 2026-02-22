from reco_trading.kernel.quant_kernel import DecisionEngine


def test_decision_engine_returns_hold_when_signal_is_weak():
    engine = DecisionEngine(buy_threshold=0.56, sell_threshold=0.44, min_edge=0.03)
    decision, score = engine.decide({'momentum': 0.52, 'mean_reversion': 0.51}, regime='range')
    assert decision == 'HOLD'
    assert 0.0 <= score <= 1.0


def test_decision_engine_returns_buy_in_trend_with_strong_scores():
    engine = DecisionEngine()
    decision, score = engine.decide({'momentum': 0.9, 'mean_reversion': 0.7}, regime='trend')
    assert decision == 'BUY'
    assert score > 0.56
