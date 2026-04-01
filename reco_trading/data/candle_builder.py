from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

_OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def ohlcv_to_frame(ohlcv: Iterable[Sequence[object]]) -> pd.DataFrame:
    """Convert raw OHLCV rows into a validated pandas DataFrame."""
    rows: list[dict[str, float | int]] = []
    for candle in ohlcv:
        if len(candle) < 6:
            continue
        try:
            timestamp = int(candle[0])
            open_price = float(candle[1])
            high_price = float(candle[2])
            low_price = float(candle[3])
            close_price = float(candle[4])
            volume = float(candle[5])
        except (TypeError, ValueError):
            continue

        if high_price < max(open_price, close_price):
            continue
        if low_price > min(open_price, close_price):
            continue
        if low_price > high_price:
            continue

        rows.append(
            {
                "timestamp": timestamp,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
            }
        )

    frame = pd.DataFrame(rows, columns=_OHLCV_COLUMNS)
    if frame.empty:
        return frame

    frame = frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame = frame.reset_index(drop=True)
    return frame
