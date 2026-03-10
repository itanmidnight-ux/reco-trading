from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import pandas as pd


@dataclass(slots=True)
class HistoricalDataset:
    """Container for normalized historical OHLCV data."""

    frame: pd.DataFrame
    source: str


class BacktestDataLoader:
    """Loads and validates historical candle data for backtesting."""

    REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def from_csv(self, path: str | Path) -> HistoricalDataset:
        frame = pd.read_csv(path)
        return self._normalize(frame, source=str(path))

    def from_dataframe(self, frame: pd.DataFrame, source: str = "dataframe") -> HistoricalDataset:
        return self._normalize(frame.copy(), source=source)

    def _normalize(self, frame: pd.DataFrame, source: str) -> HistoricalDataset:
        missing = [c for c in self.REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"missing_required_columns={missing}")

        for col in self.REQUIRED_COLUMNS:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=list(self.REQUIRED_COLUMNS))
        frame = frame[(frame["open"] > 0) & (frame["high"] > 0) & (frame["low"] > 0) & (frame["close"] > 0)]
        frame = frame[(frame["high"] >= frame[["open", "close", "low"]].max(axis=1))]
        frame = frame[(frame["low"] <= frame[["open", "close", "high"]].min(axis=1))]
        frame = frame[frame["volume"] >= 0]
        frame = frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        return HistoricalDataset(frame=frame, source=source)
