from __future__ import annotations

import pandas as pd

from reco_trading.strategy.indicators import apply_indicators


def _frame(rows: int = 120) -> pd.DataFrame:
    idx = pd.RangeIndex(rows)
    close = pd.Series([100 + (i * 0.2) + ((-1) ** i) * 0.1 for i in range(rows)], index=idx)
    open_ = close - 0.15
    high = close + 0.4
    low = close - 0.5
    volume = pd.Series([1000 + (i % 7) * 20 for i in range(rows)], index=idx)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def test_apply_indicators_adds_candle_structure_columns() -> None:
    df_result = apply_indicators(_frame())
    required_columns = [
        "body_size_abs",
        "body_ratio",
        "upper_wick",
        "lower_wick",
        "wick_ratio",
        "is_pin_bar",
        "is_hammer",
        "engulfing_bull",
        "engulfing_bear",
        "is_doji",
        "close_location",
    ]

    for col in required_columns:
        assert col in df_result.columns

    last_row = df_result.iloc[-1]
    assert 0 <= last_row["close_location"] <= 1
    assert pd.api.types.is_bool_dtype(df_result["is_pin_bar"])
    assert df_result.isnull().sum().sum() == 0
    assert pd.api.types.is_float_dtype(df_result["body_ratio"])
