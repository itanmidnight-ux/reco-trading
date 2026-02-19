from __future__ import annotations

from dataclasses import dataclass

from trading_system.app.services.feature_engineering.pipeline import FeatureVector


@dataclass
class RegimeState:
    name: str
    weights: dict[str, float]


class RegimeDetectionService:
    def detect(self, f: FeatureVector) -> RegimeState:
        if f.volatility > 1.5:
            return RegimeState('HighVolatility', {'rf': 0.2, 'xgb': 0.3, 'lstm': 0.5})
        if f.hh_hl_lh_ll > 0 and f.ema9 > f.ema21 > f.ema50:
            return RegimeState('Trend_Bull', {'rf': 0.25, 'xgb': 0.45, 'lstm': 0.3})
        if f.hh_hl_lh_ll < 0 and f.ema9 < f.ema21 < f.ema50:
            return RegimeState('Trend_Bear', {'rf': 0.25, 'xgb': 0.45, 'lstm': 0.3})
        if abs(f.kurtosis) > 8:
            return RegimeState('ExtremeEvent', {'rf': 0.1, 'xgb': 0.25, 'lstm': 0.65})
        return RegimeState('Range', {'rf': 0.45, 'xgb': 0.35, 'lstm': 0.2})
