from __future__ import annotations

import pandas as pd

from reco_trading.strategy.signal_engine import SignalEngine


def _frame(*, close: float, ema20: float, ema50: float, rsi: float, volume: float = 1200.0, vol_ma20: float = 1000.0, atr: float = 1.0, adx: float = 20.0, rows: int = 30) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "open": [close - 0.2] * rows,
            "high": [close + 0.3] * rows,
            "low": [close - 0.4] * rows,
            "close": [close] * rows,
            "ema20": [ema20] * rows,
            "ema50": [ema50] * rows,
            "rsi": [rsi] * rows,
            "volume": [volume] * rows,
            "vol_ma20": [vol_ma20] * rows,
            "atr": [atr] * rows,
            "adx": [adx] * rows,
            "macd_diff": [0.1] * rows,
            "stoch_k": [55.0] * rows,
        }
    )
    base.loc[rows - 2, ["high", "low", "close"]] = [close + 0.2, close - 0.5, close - 0.1]
    base.loc[rows - 1, ["high", "low", "close"]] = [close + 0.4, close - 0.3, close + 0.2]
    return base


def test_signal_engine_sets_neutral_trend_when_timeframes_disagree() -> None:
    engine = SignalEngine()
    df5m = _frame(close=101.0, ema20=102.0, ema50=100.0, rsi=58.0)
    df15m = _frame(close=101.0, ema20=99.0, ema50=100.0, rsi=45.0)

    result = engine.generate(df5m, df15m)

    assert result.trend == "NEUTRAL"
