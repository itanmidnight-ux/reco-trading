from __future__ import annotations

import asyncio

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from reco_trading.ai.order_flow_transformer import (
    OrderBookSnapshot,
    OrderFlowInferenceAdapter,
    OrderFlowTransformer,
)
from reco_trading.ai.rl_agent import TradingRLAgent
from reco_trading.core.signal_fusion import SignalFusionEngine


def test_order_book_snapshot_contract_to_vector() -> None:
    snapshot = OrderBookSnapshot(
        book_features=[1.0, 2.0, 3.0, 4.0],
        obi=0.2,
        cvd=0.1,
        spread=0.0008,
        trade_print_features=[0.7, 120.0],
    )
    vec = snapshot.to_feature_vector()
    assert len(vec) == 9
    assert vec[4] == 0.2


def test_transformer_forward_returns_probability() -> None:
    model = OrderFlowTransformer(input_dim=9, model_dim=32, num_heads=4, num_layers=2, ff_dim=64)
    x = torch.randn(2, 16, 9)
    probs = model(x)
    assert probs.shape == (2,)
    assert torch.all((probs >= 0.0) & (probs <= 1.0))


def test_async_inference_adapter_returns_transformer_prob_up() -> None:
    model = OrderFlowTransformer(input_dim=9, model_dim=32, num_heads=4, num_layers=1, ff_dim=64)
    adapter = OrderFlowInferenceAdapter(model=model, latency_budget_ms=100.0)
    seq = np.random.randn(16, 9).astype(np.float32)

    result = asyncio.run(adapter.infer_transformer_prob_up(seq))

    assert 0.0 <= result.transformer_prob_up <= 1.0
    assert result.timed_out is False


def test_signal_fusion_accepts_transformer_probability() -> None:
    engine = SignalFusionEngine()
    engine.update_performance("momentum", 0.01)
    engine.update_performance("mean_reversion", 0.005)

    out = engine.fuse(
        signals={"momentum": 0.2, "mean_reversion": -0.1},
        regime="trend",
        volatility=0.02,
        transformer_prob_up=0.9,
    )
    assert 0.0 <= out <= 1.0


def test_rl_agent_state_includes_transformer_probability() -> None:
    agent = TradingRLAgent(redis_url="redis://localhost:6379/15")
    state = {
        "volatility": 0.2,
        "regime": "trend",
        "win_rate": 0.55,
        "drawdown": 0.01,
        "sharpe": 0.5,
        "obi": 0.1,
        "spread": 0.0007,
        "transformer_prob_up": 0.82,
    }
    key = agent._state_key(state)
    assert key.count("|") == 7
