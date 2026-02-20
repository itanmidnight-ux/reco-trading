from __future__ import annotations

from reco_trading.hft.adaptive_market_maker import AdaptiveMarketMaker, MarketMakingState


def test_base_quote_uses_mid_plus_minus_spread_factor_atr() -> None:
    state = MarketMakingState(risk_aversion_gamma=0.0)
    mm = AdaptiveMarketMaker(state=state, spread_factor=2.0)

    decision = mm.compute_quotes(
        mid_price=100.0,
        atr=0.5,
        volatility=0.01,
        sigma=0.0,
        time_horizon=1.0,
        vpin=0.1,
        liquidity_shock=False,
    )

    assert decision.bid == 99.0
    assert decision.ask == 101.0


def test_avellaneda_stoikov_spread_and_reservation_price_inventory_adjustment() -> None:
    state = MarketMakingState(inventory=50.0, max_inventory=100.0, risk_aversion_gamma=0.2)
    mm = AdaptiveMarketMaker(state=state, spread_factor=0.0, inventory_risk_coefficient=1.0)

    decision = mm.compute_quotes(
        mid_price=100.0,
        atr=0.0,
        volatility=0.01,
        sigma=0.5,
        time_horizon=2.0,
        vpin=0.0,
        liquidity_shock=False,
    )

    sigma_sq_t = 0.5**2 * 2.0
    expected_optimal = 0.2 * sigma_sq_t
    expected_reservation = 100.0 - (0.2 * sigma_sq_t * 0.5)

    assert abs(decision.reservation_price - expected_reservation) < 1e-12
    assert abs((decision.ask - decision.bid) - 2 * expected_optimal) < 1e-12


def test_dynamic_spread_widens_for_volatility_shock_and_vpin() -> None:
    state = MarketMakingState(risk_aversion_gamma=0.0)
    mm = AdaptiveMarketMaker(
        state=state,
        spread_factor=1.0,
        volatility_high_threshold=0.02,
        vpin_threshold=0.6,
        high_vol_multiplier=1.4,
        liquidity_shock_multiplier=1.5,
        vpin_multiplier=1.2,
    )

    baseline = mm.compute_quotes(
        mid_price=100.0,
        atr=1.0,
        volatility=0.01,
        sigma=0.0,
        time_horizon=1.0,
        vpin=0.2,
        liquidity_shock=False,
    )
    stressed = mm.compute_quotes(
        mid_price=100.0,
        atr=1.0,
        volatility=0.05,
        sigma=0.0,
        time_horizon=1.0,
        vpin=0.9,
        liquidity_shock=True,
    )

    assert stressed.spread > baseline.spread


def test_inventory_neutralization_biases_quotes_and_reduces_size() -> None:
    state = MarketMakingState(
        inventory=90.0,
        max_inventory=100.0,
        inventory_neutralization_ratio=0.7,
        base_order_size=10.0,
    )
    mm = AdaptiveMarketMaker(state=state, spread_factor=0.0)

    decision = mm.compute_quotes(
        mid_price=100.0,
        atr=0.0,
        volatility=0.01,
        sigma=0.0,
        time_horizon=1.0,
        vpin=0.0,
        liquidity_shock=False,
    )

    assert decision.should_unwind is True
    assert decision.bid_size < state.base_order_size
    assert decision.ask_size >= state.base_order_size
    assert decision.unwind_size > 0.0


def test_transformer_probability_biases_reservation_price() -> None:
    state = MarketMakingState(risk_aversion_gamma=0.0)
    mm = AdaptiveMarketMaker(state=state, spread_factor=0.0)

    up = mm.compute_quotes(
        mid_price=100.0,
        atr=0.0,
        volatility=0.01,
        sigma=0.0,
        time_horizon=1.0,
        vpin=0.0,
        liquidity_shock=False,
        transformer_prob_up=0.9,
    )
    down = mm.compute_quotes(
        mid_price=100.0,
        atr=0.0,
        volatility=0.01,
        sigma=0.0,
        time_horizon=1.0,
        vpin=0.0,
        liquidity_shock=False,
        transformer_prob_up=0.1,
    )

    assert up.reservation_price > down.reservation_price
