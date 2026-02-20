from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(slots=True)
class MicrostructureSnapshot:
    """Snapshot de señales de microestructura para consumo downstream."""

    obi: float
    cvd: float
    spread: float
    vpin: float
    liquidity_shock: bool


class OrderBookMicrostructureAnalyzer:
    """Analizador institucional de order book.

    Calcula OBI, CVD rolling, spread dinámico, VPIN simplificado y shock de liquidez.
    """

    def __init__(
        self,
        depth_levels: int = 10,
        cvd_window: int = 200,
        vpin_buckets: int = 24,
        liquidity_zscore_threshold: float = -2.5,
        spread_jump_threshold: float = 0.35,
    ) -> None:
        self.depth_levels = max(depth_levels, 1)
        self.cvd_window = max(cvd_window, 5)
        self.vpin_buckets = max(vpin_buckets, 6)
        self.liquidity_zscore_threshold = float(liquidity_zscore_threshold)
        self.spread_jump_threshold = float(spread_jump_threshold)

        self._cvd_history: deque[float] = deque(maxlen=self.cvd_window)
        self._depth_history: deque[float] = deque(maxlen=self.cvd_window)
        self._spread_history: deque[float] = deque(maxlen=self.cvd_window)
        self._volume_imbalance_history: deque[float] = deque(maxlen=self.vpin_buckets)

    def _validate_depth(self, bids: Iterable[tuple[float, float]], asks: Iterable[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
        bids_arr = np.asarray(list(bids), dtype=float)
        asks_arr = np.asarray(list(asks), dtype=float)
        if bids_arr.ndim != 2 or asks_arr.ndim != 2 or bids_arr.shape[1] < 2 or asks_arr.shape[1] < 2:
            raise ValueError("Order book corrupto: formato inválido")
        if bids_arr.shape[0] < self.depth_levels or asks_arr.shape[0] < self.depth_levels:
            raise ValueError("Profundidad insuficiente para análisis microestructural")
        if np.any(bids_arr[:, 0] <= 0.0) or np.any(asks_arr[:, 0] <= 0.0):
            raise ValueError("Order book corrupto: precios no positivos")
        if np.any(bids_arr[:, 1] < 0.0) or np.any(asks_arr[:, 1] < 0.0):
            raise ValueError("Order book corrupto: volúmenes negativos")
        return bids_arr[: self.depth_levels], asks_arr[: self.depth_levels]

    def compute(self, bids: Iterable[tuple[float, float]], asks: Iterable[tuple[float, float]]) -> MicrostructureSnapshot:
        bids_arr, asks_arr = self._validate_depth(bids, asks)

        bid_volume = float(bids_arr[:, 1].sum())
        ask_volume = float(asks_arr[:, 1].sum())
        volume_total = max(bid_volume + ask_volume, 1e-9)

        obi = float((bid_volume - ask_volume) / volume_total)
        delta = float(ask_volume - bid_volume)
        next_cvd = (self._cvd_history[-1] if self._cvd_history else 0.0) + delta
        self._cvd_history.append(next_cvd)

        best_bid = float(bids_arr[0, 0])
        best_ask = float(asks_arr[0, 0])
        mid = max((best_bid + best_ask) / 2.0, 1e-9)
        spread = float(max(best_ask - best_bid, 0.0) / mid)

        self._spread_history.append(spread)

        imbalance = abs(delta) / volume_total
        self._volume_imbalance_history.append(float(imbalance))
        vpin = float(np.mean(self._volume_imbalance_history)) if self._volume_imbalance_history else 0.0

        depth_total = bid_volume + ask_volume
        self._depth_history.append(depth_total)
        depth_arr = np.asarray(list(self._depth_history), dtype=float)
        spread_arr = np.asarray(list(self._spread_history), dtype=float)
        depth_zscore = 0.0
        spread_jump = 0.0
        if depth_arr.size >= 10:
            depth_std = float(depth_arr.std()) + 1e-9
            depth_zscore = float((depth_arr[-1] - depth_arr.mean()) / depth_std)
        if spread_arr.size >= 10:
            baseline = float(np.median(spread_arr[:-1])) if spread_arr.size > 1 else spread_arr[-1]
            spread_jump = float((spread_arr[-1] - baseline) / (baseline + 1e-9))

        liquidity_shock = depth_zscore <= self.liquidity_zscore_threshold or spread_jump >= self.spread_jump_threshold

        return MicrostructureSnapshot(
            obi=obi,
            cvd=float(self._cvd_history[-1]),
            spread=spread,
            vpin=vpin,
            liquidity_shock=bool(liquidity_shock),
        )
