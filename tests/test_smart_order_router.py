from reco_trading.execution.smart_order_router import (
    ImpactModel,
    OrderSplitter,
    SmartOrderRouter,
    VenueScoreModel,
    VenueSnapshot,
)


def test_venue_score_model_applies_latency_tiebreak() -> None:
    model = VenueScoreModel()
    venues = [
        VenueSnapshot('a', spread_bps=10, depth=100, latency_ms=40, fee_bps=8, fill_ratio=0.95, liquidity=500),
        VenueSnapshot('b', spread_bps=10, depth=100, latency_ms=10, fee_bps=8, fill_ratio=0.95, liquidity=500),
    ]
    scores = model.score(venues, epsilon=0.2)
    assert scores['b'] > scores['a']


def test_order_splitter_supports_twap_vwap_iceberg() -> None:
    splitter = OrderSplitter()
    assert len(splitter.split(100, 'TWAP', slices=4)) == 4
    vwap = splitter.split(100, 'VWAP', slices=4, expected_volume_profile=[1, 2, 3, 4])
    assert round(sum(vwap), 10) == 100
    iceberg = splitter.split(100, 'ICEBERG', slices=5, iceberg_peak_ratio=0.2)
    assert all(chunk <= 20 for chunk in iceberg)


def test_impact_model_formula() -> None:
    model = ImpactModel(lambda_impact=0.7)
    assert model.estimate(order_size=50, liquidity=200) == 0.175


def test_sor_routes_children_with_allocations() -> None:
    sor = SmartOrderRouter(epsilon=0.05)
    venues = [
        VenueSnapshot('v1', spread_bps=11, depth=120, latency_ms=35, fee_bps=9, fill_ratio=0.97, liquidity=200),
        VenueSnapshot('v2', spread_bps=10, depth=90, latency_ms=18, fee_bps=11, fill_ratio=0.96, liquidity=150),
    ]
    routed = sor.route_order(amount=100, venues=venues, strategy='TWAP', slices=4)
    assert routed
    assert round(sum(float(x['amount']) for x in routed), 8) == 100
