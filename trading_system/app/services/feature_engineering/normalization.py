from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NormalizationResult:
    values: dict[str, float]


class FeatureNormalizer:
    def __init__(self) -> None:
        self._ranges = {
            'rsi': (0, 100),
            'orderbook_imbalance': (-1, 1),
            'volatility': (0, 5),
            'zscore_price': (-4, 4),
            'stat_confidence': (0, 1),
        }

    def normalize(self, features: dict[str, float]) -> NormalizationResult:
        out: dict[str, float] = {}
        for key, value in features.items():
            if key in self._ranges:
                low, high = self._ranges[key]
                span = high - low
                out[key] = 0.5 if span == 0 else max(0.0, min(1.0, (value - low) / span))
            else:
                out[key] = value
        return NormalizationResult(values=out)
