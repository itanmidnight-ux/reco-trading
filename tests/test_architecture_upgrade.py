import numpy as np
import pandas as pd

from reco_trading.adaptive.frequency_controller import FrequencyController
from reco_trading.portfolio.allocator import PortfolioAllocator
from reco_trading.portfolio.exposure_model import ExposureModel
from reco_trading.statistics.drift_detector import CUSUMDrift, KLDivergenceDrift


def test_portfolio_allocator_single_asset_weight_one():
    allocator = PortfolioAllocator()
    weights = allocator.allocate({'BTCUSDT': 0.01}, {'BTCUSDT': 0.02}, None)
    assert weights == {'BTCUSDT': 1.0}


def test_exposure_model_returns_correlation_matrix():
    model = ExposureModel(lookback_bars=50)
    prices = {
        'A': pd.Series(np.linspace(100, 120, 80)),
        'B': pd.Series(np.linspace(200, 230, 80)),
    }
    corr = model.compute_correlation_matrix(prices)
    assert set(corr.columns) == {'A', 'B'}


def test_drift_detectors_emit_finite_scores():
    cusum = CUSUMDrift(k=0.001, h=0.01)
    score = 0.0
    for value in [0.0, 0.001, 0.002, 0.01]:
        score = cusum.update(value)
    assert 0.0 <= score <= 1.0

    kl = KLDivergenceDrift(bins=10)
    recent = np.random.normal(0.2, 1.0, 200)
    historical = np.random.normal(0.0, 1.0, 200)
    kl_value = kl.compute(recent, historical)
    assert np.isfinite(kl_value)


def test_frequency_controller_clamps_to_friction_cost():
    controller = FrequencyController(target_trades_per_day=10, adjustment_strength=0.1)
    adjusted = controller.adjust_threshold(dynamic_edge_threshold=0.001, friction_cost=0.002)
    assert adjusted >= 0.002


def test_frequency_controller_deduplicates_trade_ids():
    controller = FrequencyController(target_trades_per_day=10, adjustment_strength=0.1)
    controller.register_trade(trade_id='abc123')
    controller.register_trade(trade_id='abc123')
    assert len(controller.trade_timestamps) == 1
