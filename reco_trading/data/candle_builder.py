from __future__ import annotations

import pandas as pd


def ohlcv_to_frame(ohlcv: list[list[float]]) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if df.empty:
        return df

    numeric_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols)
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp")

    valid = (
        (df["high"] >= df[["open", "close"]].max(axis=1))
        & (df["low"] <= df[["open", "close"]].min(axis=1))
        & (df["high"] >= df["low"])
        & (df[["open", "high", "low", "close", "volume"]] >= 0).all(axis=1)
    )
    df = df.loc[valid].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.reset_index(drop=True)
