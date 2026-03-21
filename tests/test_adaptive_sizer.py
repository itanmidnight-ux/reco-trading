from reco_trading.risk.adaptive_sizer import AdaptiveSizer


def _compute(**overrides):
    sizer = overrides.pop(
        "sizer",
        AdaptiveSizer(base_risk_fraction=0.01, min_multiplier=0.30, max_multiplier=1.50),
    )
    params = {
        "equity": 1000.0,
        "price": 100.0,
        "stop_loss": 95.0,
        "atr": 4.0,
        "confidence": 0.70,
        "recent_pnls": [1.0, -1.0, 0.5],
        "volatility_multiplier": 1.0,
    }
    params.update(overrides)
    return sizer.compute(**params)


def test_higher_confidence_produces_larger_size() -> None:
    lower = _compute(confidence=0.60)
    higher = _compute(confidence=0.95)

    assert higher.quantity > lower.quantity
    assert higher.size_multiplier > lower.size_multiplier


def test_losing_streak_reduces_size() -> None:
    neutral = _compute(recent_pnls=[1.0, -1.0, 1.0])
    losing = _compute(recent_pnls=[-1.0, -2.0, -3.0])

    assert losing.quantity < neutral.quantity
    assert losing.size_multiplier < neutral.size_multiplier


def test_size_multiplier_never_exceeds_max_multiplier() -> None:
    sizer = AdaptiveSizer(base_risk_fraction=0.01, min_multiplier=0.30, max_multiplier=1.10)
    decision = _compute(sizer=sizer, confidence=1.0, recent_pnls=[2.0, 2.0, 2.0], volatility_multiplier=2.0)

    assert decision.size_multiplier == 1.10


def test_size_multiplier_never_drops_below_min_multiplier() -> None:
    sizer = AdaptiveSizer(base_risk_fraction=0.01, min_multiplier=0.40, max_multiplier=1.50)
    decision = _compute(sizer=sizer, confidence=0.0, recent_pnls=[-1.0, -1.0, -1.0], volatility_multiplier=0.01)

    assert decision.size_multiplier == 0.40


def test_zero_equity_returns_zero_quantity() -> None:
    decision = _compute(equity=0.0)

    assert decision.quantity == 0.0
    assert decision.risk_amount == 0.0
