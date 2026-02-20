from __future__ import annotations

import pandas as pd

from reco_trading.ai.rl_agent import TradingRLAgent
from reco_trading.core.meta_learning import AdaptiveMetaLearner
from reco_trading.core.microstructure import OrderBookMicrostructureAnalyzer
from reco_trading.core.portfolio_optimization import ConvexPortfolioOptimizer


def test_microstructure_snapshot_computation() -> None:
    analyzer = OrderBookMicrostructureAnalyzer(depth_levels=3, cvd_window=20, vpin_buckets=8)
    bids = [(100.0, 2.0), (99.9, 1.5), (99.8, 1.2)]
    asks = [(100.1, 1.8), (100.2, 1.1), (100.3, 1.0)]
    snap = analyzer.compute(bids, asks)

    assert -1.0 <= snap.obi <= 1.0
    assert snap.spread > 0.0
    assert 0.0 <= snap.vpin <= 1.0


def test_meta_learning_weights_normalized() -> None:
    learner = AdaptiveMetaLearner(model_names=["m1", "m2"], window=50)
    for _ in range(20):
        learner.register_observation("m1", True, "trend")
        learner.register_observation("m2", False, "trend")
    out = learner.optimize(regime="trend", volatility=0.01, drawdown=0.03)
    assert abs(sum(out.model_weights.values()) - 1.0) < 1e-6
    assert out.model_weights["m1"] > out.model_weights["m2"]


def test_portfolio_optimizer_mean_variance() -> None:
    returns = pd.DataFrame(
        {
            "BTCUSDT": [0.001, 0.002, -0.001, 0.003, 0.0005],
            "ETHUSDT": [0.0012, 0.0018, -0.0008, 0.0022, 0.0006],
        }
    )
    optimizer = ConvexPortfolioOptimizer()
    result = optimizer.mean_variance(returns, target_return=0.0005, exposure_limit=1.0)
    assert abs(sum(result.weights.values()) - 1.0) < 1e-5
    assert result.expected_volatility >= 0.0


def test_rl_agent_select_action_and_update() -> None:
    agent = TradingRLAgent(redis_url="redis://localhost:6379/15")
    state = {
        "volatility": 0.3,
        "regime": "trend",
        "win_rate": 0.55,
        "drawdown": 0.02,
        "sharpe": 0.8,
        "obi": 0.1,
        "spread": 0.0008,
    }
    action = agent.select_action(state)
    assert action.size_multiplier >= 0.0
    agent.update_policy(state, delta_equity=0.001, drawdown=0.02)
