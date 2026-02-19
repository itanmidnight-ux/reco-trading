from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque


@dataclass
class OhlcvState:
    close: deque[float] = field(default_factory=lambda: deque(maxlen=3000))
    high: deque[float] = field(default_factory=lambda: deque(maxlen=3000))
    low: deque[float] = field(default_factory=lambda: deque(maxlen=3000))
    volume: deque[float] = field(default_factory=lambda: deque(maxlen=3000))
    bid_qty: float = 0.0
    ask_qty: float = 0.0

    def ingest_kline(self, payload: dict) -> None:
        k = payload.get('k', {})
        if not k or not k.get('x'):
            return
        self.close.append(float(k['c']))
        self.high.append(float(k['h']))
        self.low.append(float(k['l']))
        self.volume.append(float(k['v']))

    def ingest_depth(self, payload: dict) -> None:
        self.bid_qty = sum(float(q) for _, q in payload.get('b', [])[:10])
        self.ask_qty = sum(float(q) for _, q in payload.get('a', [])[:10])
