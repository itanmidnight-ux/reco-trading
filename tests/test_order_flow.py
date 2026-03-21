from __future__ import annotations

import pandas as pd

from reco_trading.strategy.order_flow import OrderFlowAnalyzer


def test_order_flow_weighted_magnitude() -> None:
    """Verifica que una barra bajista grande domina barras alcistas pequeñas."""
    data = [
        (100.0, 100.5, 99.5, 100.2, 1000.0, 1.0),
        (100.0, 100.5, 99.5, 100.2, 1000.0, 1.0),
        (100.0, 100.5, 99.5, 100.2, 1000.0, 1.0),
        (100.0, 100.2, 97.5, 98.0, 2000.0, 1.0),
    ]
    df = pd.DataFrame(data, columns=["open", "high", "low", "close", "volume", "atr"])

    result = OrderFlowAnalyzer(lookback=10).evaluate(df)

    assert result.signal == "SELL"
    assert 0.0 <= result.buy_pressure <= 1.0
    assert 0.0 <= result.sell_pressure <= 1.0
    assert abs(result.buy_pressure + result.sell_pressure - 1.0) < 0.001


def test_order_flow_handles_empty_frame() -> None:
    analyzer = OrderFlowAnalyzer()
    result = analyzer.evaluate(pd.DataFrame())
    assert result.signal == "NEUTRAL"
    assert result.buy_pressure == 0.5
    assert result.sell_pressure == 0.5
