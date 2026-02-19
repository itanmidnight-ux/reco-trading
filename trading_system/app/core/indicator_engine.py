from __future__ import annotations

from dataclasses import dataclass

from trading_system.app.services.feature_engineering.pipeline import FeatureVector


@dataclass
class IndicatorSnapshot:
    rsi: float
    macd_hist: float
    ema20_gt_50: bool
    ema50_gt_200_proxy: bool
    atr: float
    adx_proxy: float
    bollinger_zscore: float


class IndicatorEngine:
    def extract(self, f: FeatureVector) -> IndicatorSnapshot:
        ema20_gt_50 = f.ema9 > f.ema21
        ema50_gt_200_proxy = f.ema21 > f.ema50
        adx_proxy = min(100.0, max(0.0, abs(f.orderbook_imbalance) * 200))
        return IndicatorSnapshot(
            rsi=f.rsi,
            macd_hist=f.macd_hist,
            ema20_gt_50=ema20_gt_50,
            ema50_gt_200_proxy=ema50_gt_200_proxy,
            atr=f.atr,
            adx_proxy=adx_proxy,
            bollinger_zscore=f.bollinger_zscore,
        )
