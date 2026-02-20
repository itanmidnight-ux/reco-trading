import pytest

from reco_trading.research.dark_pool_simulator import DarkPoolSimulator


def test_hidden_fill_probability_responds_to_tif_and_volatility():
    sim = DarkPoolSimulator()

    low_tif = sim.hidden_fill_probability(
        order_qty=1000,
        visible_volume=800,
        volatility=0.01,
        time_in_force_seconds=10,
    )
    high_tif = sim.hidden_fill_probability(
        order_qty=1000,
        visible_volume=800,
        volatility=0.01,
        time_in_force_seconds=180,
    )
    high_vol = sim.hidden_fill_probability(
        order_qty=1000,
        visible_volume=800,
        volatility=0.08,
        time_in_force_seconds=180,
    )

    assert high_tif > low_tif
    assert high_vol < high_tif


def test_compare_routes_exposes_sor_api_and_metrics_history():
    sim = DarkPoolSimulator()
    routes = sim.compare_routes(
        order_qty=1200,
        side="BUY",
        mid_price=100.0,
        visible_volume=950,
        volatility=0.015,
        spread_bps=6.0,
        time_in_force_seconds=120,
        hybrid_dark_ratio=0.6,
        seed=7,
    )

    assert set(routes) == {"visible-only", "dark-only", "hybrid-split"}
    for result in routes.values():
        assert 0.0 <= result.fill_ratio <= 1.0
        assert result.filled_qty <= result.requested_qty + 1e-6
        assert result.avg_confirmation_latency_ms >= 0.0

    history = sim.metrics_history()
    assert len(history) == 3
    assert {"route", "slippage_bps", "fill_ratio", "adverse_selection_cost_bps"}.issubset(history.columns)


def test_dark_route_has_higher_latency_than_visible_route_on_average():
    sim = DarkPoolSimulator()
    routes = sim.compare_routes(
        order_qty=800,
        side="SELL",
        mid_price=50.0,
        visible_volume=900,
        volatility=0.02,
        spread_bps=8.0,
        time_in_force_seconds=90,
        seed=21,
    )

    assert routes["dark-only"].avg_confirmation_latency_ms > routes["visible-only"].avg_confirmation_latency_ms
