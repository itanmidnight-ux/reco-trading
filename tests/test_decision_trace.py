from __future__ import annotations

from reco_trading.strategy.confidence_model import ConfidenceModel
from reco_trading.strategy.signal_engine import SignalBundle


def test_confidence_model_explain_contains_factor_scores_and_threshold() -> None:
    model = ConfidenceModel()
    bundle = SignalBundle(
        trend="BUY",
        momentum="BUY",
        volume="NEUTRAL",
        volatility="BUY",
        structure="SELL",
        order_flow="BUY",
        regime="NORMAL_VOLATILITY",
        regime_trade_allowed=True,
        size_multiplier=1.0,
        atr_ratio=0.01,
    )
    explained = model.explain(bundle, trade_threshold=0.5)
    assert explained["side"] in {"BUY", "SELL", "HOLD"}
    assert isinstance(explained["factor_scores"], dict)
    assert explained["threshold"] == 0.5
