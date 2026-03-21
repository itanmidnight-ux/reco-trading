from __future__ import annotations

import pandas as pd

from reco_trading.strategy.signal_engine import SignalEngine


def _frame(rows: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0][:rows],
            "high": [101.0, 102.0, 103.0][:rows],
            "low": [99.0, 100.0, 101.0][:rows],
            "close": [100.5, 101.5, 102.5][:rows],
            "volume": [1000.0, 1200.0, 1400.0][:rows],
            "ema20": [100.0, 101.0, 102.0][:rows],
            "ema50": [100.0, 100.5, 101.0][:rows],
            "rsi": [50.0, 56.0, 58.0][:rows],
            "atr": [1.0, 1.0, 1.0][:rows],
            "adx": [20.0, 22.0, 24.0][:rows],
            "vol_ma20": [1000.0, 1100.0, 1200.0][:rows],
        }
    )


def test_signal_engine_returns_neutral_bundle_when_frames_are_too_short() -> None:
    engine = SignalEngine()
    bundle = engine.generate(_frame(rows=1), _frame(rows=1))
    assert bundle.regime_trade_allowed is False
    assert bundle.trend == "NEUTRAL"
    assert bundle.order_flow == "NEUTRAL"


def test_signal_engine_trend_is_neutral_when_timeframes_disagree() -> None:
    engine = SignalEngine()
    df5 = _frame()
    df15 = _frame()
    df15.loc[df15.index[-1], "ema20"] = 100.0
    df15.loc[df15.index[-1], "ema50"] = 101.0

    bundle = engine.generate(df5, df15)

    assert bundle.trend == "NEUTRAL"
