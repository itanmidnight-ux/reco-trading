from __future__ import annotations

from dataclasses import dataclass

from trading_system.app.services.feature_engineering.indicators import (
    atr,
    bollinger_zscore,
    ema,
    kurtosis,
    macd_hist,
    rolling_vol,
    rsi,
    skewness,
    sma,
    zscore,
)
from trading_system.app.services.market_data.history_builder import OhlcvState
from trading_system.app.services.feature_engineering.statistics import statistical_confirmation


@dataclass
class FeatureVector:
    rsi: float
    ema9: float
    ema21: float
    ema50: float
    macd_hist: float
    atr: float
    bollinger_zscore: float
    orderbook_imbalance: float
    delta_volume: float
    spread_proxy: float
    volatility: float
    zscore_price: float
    skewness: float
    kurtosis: float
    hh_hl_lh_ll: float
    breakout_score: float
    consolidation_score: float
    stat_t: float
    stat_pvalue: float
    hurst_proxy: float
    stationarity_proxy: float
    stat_confidence: float


class FeatureEngineeringService:
    def build(self, state: OhlcvState) -> FeatureVector:
        c = list(state.close)
        h = list(state.high)
        l = list(state.low)
        v = list(state.volume)
        total_depth = state.bid_qty + state.ask_qty
        imbalance = ((state.bid_qty - state.ask_qty) / total_depth) if total_depth > 0 else 0.0
        delta_v = (v[-1] - v[-2]) if len(v) > 2 else 0.0
        spread_proxy = abs(imbalance) * 0.001

        structure = 0.0
        if len(c) > 3:
            structure = 1.0 if c[-1] > c[-2] > c[-3] else -1.0 if c[-1] < c[-2] < c[-3] else 0.0
        breakout = 1.0 if len(c) > 30 and c[-1] > max(c[-30:-1]) else 0.0
        consolidation = 1.0 if rolling_vol(c, 20) < 0.3 else 0.0

        stats = statistical_confirmation(c)

        return FeatureVector(
            rsi=rsi(c),
            ema9=ema(c, 9),
            ema21=ema(c, 21),
            ema50=ema(c, 50),
            macd_hist=macd_hist(c) if len(c) > 40 else 0.0,
            atr=atr(h, l, c),
            bollinger_zscore=bollinger_zscore(c),
            orderbook_imbalance=imbalance,
            delta_volume=delta_v,
            spread_proxy=spread_proxy,
            volatility=rolling_vol(c),
            zscore_price=zscore(c),
            skewness=skewness(c),
            kurtosis=kurtosis(c),
            hh_hl_lh_ll=structure,
            breakout_score=breakout,
            consolidation_score=consolidation,
            stat_t=stats.t_stat,
            stat_pvalue=stats.trend_pseudo_pvalue,
            hurst_proxy=stats.hurst_proxy,
            stationarity_proxy=stats.stationarity_proxy,
            stat_confidence=stats.confidence,
        )
