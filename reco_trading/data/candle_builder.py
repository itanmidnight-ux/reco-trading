from __future__ import annotations

import pandas as pd


def ohlcv_to_frame(ohlcv: list[list[float]]) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df
