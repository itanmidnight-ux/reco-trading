from __future__ import annotations

import numpy as np
import pandas as pd

from reco_trading.core.data_buffer import MarketSnapshot
from reco_trading.core.microstructure import MicrostructureSnapshot


class FeatureEngine:
    @staticmethod
    def market_snapshot(df: pd.DataFrame, spread: float = 0.0) -> MarketSnapshot:
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        volume = df['volume'].astype(float)

        returns = np.log(close / close.shift(1)).dropna().tail(300).to_numpy(dtype=float)
        volatility = float(pd.Series(returns).tail(40).std() or 0.0)

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1] or 0.0)

        ema12 = close.ewm(span=12, adjust=False).mean().iloc[-1]
        ema26 = close.ewm(span=26, adjust=False).mean().iloc[-1]
        trend_strength = float(abs(ema12 - ema26) / max(close.iloc[-1], 1e-9))

        vwap = float((close * volume).rolling(30).sum().iloc[-1] / max(volume.rolling(30).sum().iloc[-1], 1e-9))
        vwap_distance = float((close.iloc[-1] - vwap) / max(vwap, 1e-9))

        mean20 = float(close.rolling(20).mean().iloc[-1])
        std20 = float(close.rolling(20).std().iloc[-1] or 0.0)
        bollinger_dev = float((close.iloc[-1] - mean20) / max(2 * std20, 1e-9))

        return MarketSnapshot(
            returns=returns.copy(),
            volatility=max(volatility, 0.0),
            atr=max(atr, 0.0),
            trend_strength=max(trend_strength, 0.0),
            spread=max(float(spread), 0.0),
            vwap_distance=vwap_distance,
            bollinger_deviation=bollinger_dev,
        )

    def build(self, df: pd.DataFrame, microstructure: MicrostructureSnapshot | None = None) -> pd.DataFrame:
        out = df.copy()
        out['return'] = np.log(out['close'] / out['close'].shift(1))
        out['ema12'] = out['close'].ewm(span=12, adjust=False).mean()
        out['ema26'] = out['close'].ewm(span=26, adjust=False).mean()
        out['macd'] = out['ema12'] - out['ema26']
        out['breakout20'] = out['close'] / out['high'].rolling(20).max() - 1
        out['volatility20'] = out['return'].rolling(20).std()
        out['volume_norm'] = out['volume'] / out['volume'].rolling(20).mean()

        mean20 = out['close'].rolling(20).mean()
        std20 = out['close'].rolling(20).std().replace(0, np.nan)
        out['zscore20'] = (out['close'] - mean20) / std20

        diff = out['close'].diff()
        gain = diff.clip(lower=0).rolling(14).mean()
        loss = (-diff.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
        rs = gain / loss
        out['rsi14'] = 100 - (100 / (1 + rs))

        tr = pd.concat(
            [
                out['high'] - out['low'],
                (out['high'] - out['close'].shift(1)).abs(),
                (out['low'] - out['close'].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out['atr14'] = tr.rolling(14).mean()

        out['bb_dev'] = (out['close'] - mean20) / (2 * std20)
        out['target_up'] = (out['return'].shift(-1) > 0).astype(int)
        out['target_reversion'] = (out['return'].shift(-1) * out['zscore20'] < 0).astype(int)

        if microstructure is not None:
            out['obi'] = microstructure.obi
            out['cvd'] = microstructure.cvd
            out['spread_micro'] = microstructure.spread
            out['vpin'] = microstructure.vpin
            out['liquidity_shock'] = 1.0 if microstructure.liquidity_shock else 0.0

        return out.dropna().reset_index(drop=True)
