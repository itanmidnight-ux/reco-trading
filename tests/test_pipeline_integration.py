import asyncio

import numpy as np
import pandas as pd

from scripts.live_trading import FeatureEngineAdapter, FusionEngineAdapter, RuntimeState
from reco_trading.core.feature_engine import FeatureEngine
from reco_trading.core.mean_reversion_model import MeanReversionModel
from reco_trading.core.momentum_model import MomentumModel
from reco_trading.core.pipeline import TradingPipeline


def _ohlcv(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 30000 + np.cumsum(rng.normal(0, 20, rows))
    high = close + rng.normal(5, 2, rows)
    low = close - rng.normal(5, 2, rows)
    open_ = close + rng.normal(0, 5, rows)
    volume = np.abs(rng.normal(100, 20, rows))
    ts = pd.date_range('2025-01-01', periods=rows, freq='5min', tz='UTC')
    return pd.DataFrame({'timestamp': ts, 'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume})


def test_feature_adapter_outputs_pipeline_contract():
    adapter = FeatureEngineAdapter(
        FeatureEngine(),
        MomentumModel(),
        MeanReversionModel(),
        RuntimeState(equity=1200.0, daily_pnl=0.0, consecutive_losses=0),
    )
    out = adapter.compute(_ohlcv())
    assert {'returns', 'returns_df', 'prices', 'signals', 'volatility', 'equity', 'atr', 'win_rate', 'reward_risk'}.issubset(out.keys())
    assert out['equity'] == 1200.0


def test_fusion_adapter_probability_range():
    adapter = FusionEngineAdapter()
    p = adapter.fuse({'momentum': 0.62, 'mean_reversion': 0.51}, regime='trend', volatility=0.015)
    assert 0.0 <= p <= 1.0


def test_pipeline_shutdown_is_clean():
    class _Feed:
        async def stream(self):
            for _ in range(2):
                yield _ohlcv()

    class _Feature:
        def compute(self, _data):
            return {
                'returns': np.random.normal(0, 0.01, 100),
                'returns_df': pd.DataFrame({'BTCUSDT': np.random.normal(0, 0.01, 100)}),
                'prices': pd.Series(np.linspace(10, 20, 100)),
                'signals': {'momentum': 0.8, 'mean_reversion': 0.2},
                'volatility': 0.01,
                'equity': 1000,
                'atr': 20,
                'win_rate': 0.6,
                'reward_risk': 2.0,
            }

    class _Regime:
        def predict(self, _returns, _prices):
            return {'regime': 'trend'}

    class _Fusion:
        def fuse(self, *_args, **_kwargs):
            return 0.8

    class _Risk:
        kill_switch = False

        def update_equity(self, _equity):
            return None

        def check_kill_switch(self, _equity):
            return None

        def check_correlation_risk(self, _returns_df):
            return False

        def calculate_position_size(self, **_kwargs):
            return 0.1

    class _Exec:
        async def execute(self, _side, _size):
            return None

    async def _run():
        pipeline = TradingPipeline(_Feed(), _Feature(), _Regime(), _Fusion(), _Risk(), _Exec())
        task = asyncio.create_task(pipeline.run())
        await asyncio.sleep(0.05)
        await pipeline.shutdown()
        task.cancel()

    asyncio.run(_run())
