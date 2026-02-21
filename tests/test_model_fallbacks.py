import numpy as np
import pandas as pd

from reco_trading.core.feature_engine import FeatureEngine
from reco_trading.core.mean_reversion_model import MeanReversionModel
from reco_trading.core.momentum_model import MomentumModel


def _single_regime_ohlcv(rows: int = 120) -> pd.DataFrame:
    ts = pd.date_range('2025-01-01', periods=rows, freq='5min', tz='UTC')
    close = np.full(rows, 30_000.0)
    return pd.DataFrame(
        {
            'timestamp': ts,
            'open': close,
            'high': close,
            'low': close,
            'close': close,
            'volume': np.full(rows, 100.0),
        }
    )


def test_momentum_model_returns_neutral_probability_when_only_one_class() -> None:
    frame = FeatureEngine().build(_single_regime_ohlcv())
    model = MomentumModel()
    prob = model.predict_proba_up(frame)
    assert prob == 0.5


def test_mean_reversion_model_returns_neutral_probability_when_only_one_class() -> None:
    frame = FeatureEngine().build(_single_regime_ohlcv())
    model = MeanReversionModel()
    prob = model.predict_reversion(frame)
    assert prob == 0.5
