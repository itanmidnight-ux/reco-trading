import asyncio

import pandas as pd

from reco_trading.config.settings import get_settings
from reco_trading.core.feature_engine import FeatureEngine
from reco_trading.core.fusion_engine import FusionEngine
from reco_trading.core.market_data import MarketDataService
from reco_trading.core.mean_reversion_model import MeanReversionModel
from reco_trading.core.momentum_model import MomentumModel
from reco_trading.infra.binance_client import BinanceClient
from reco_trading.research.backtest_engine import BacktestEngine
from reco_trading.research.monte_carlo import MonteCarlo
from reco_trading.research.walk_forward import WalkForward


async def main() -> None:
    s = get_settings()
    client = BinanceClient(s.binance_api_key.get_secret_value(), s.binance_api_secret.get_secret_value(), s.binance_testnet)
    frame = FeatureEngine().build(await MarketDataService(client, s.symbol, s.timeframe).latest_ohlcv(limit=1800))

    wf = WalkForward()
    engine = BacktestEngine(s.taker_fee, s.slippage_bps)
    fusion = FusionEngine()
    all_returns = []
    fold_stats = []

    for train, test in wf.generate_splits(frame):
        mm = MomentumModel()
        mr = MeanReversionModel()
        mm.fit(train)
        mr.fit(train)

        signals = []
        for i in range(len(test)):
            sub = pd.concat([train, test.iloc[: i + 1]], ignore_index=True)
            signals.append(fusion.decide(mm.predict_proba_up(sub), mr.predict_reversion(sub)))

        stats = engine.run(test.reset_index(drop=True), pd.Series(signals))
        fold_stats.append(stats)
        all_returns.extend((test['return'].shift(-1).fillna(0) * pd.Series(signals).map({'BUY': 1, 'SELL': -1, 'HOLD': 0})).tolist())

    mc = MonteCarlo().simulate(all_returns)
    print({'walk_forward_folds': fold_stats, 'monte_carlo': mc})
    await client.close()


if __name__ == '__main__':
    asyncio.run(main())
