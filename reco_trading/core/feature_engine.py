from __future__ import annotations

import numpy as np
import pandas as pd

from reco_trading.core.microstructure import MicrostructureSnapshot


class FeatureEngine:
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
