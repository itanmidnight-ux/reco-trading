from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def ohlcv_to_frame(ohlcv: list[list[float]]) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv, columns=REQUIRED_COLUMNS)
    if df.empty:
        return df

    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df[df["volume"] >= 0]
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
    df = df[(df["high"] >= df[["open", "close", "low"]].max(axis=1))]
    df = df[(df["low"] <= df[["open", "close", "high"]].min(axis=1))]

    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.reset_index(drop=True)
