import asyncio

from reco_trading.config.settings import get_settings
from reco_trading.core.feature_engine import FeatureEngine
from reco_trading.core.market_data import MarketDataService
from reco_trading.core.mean_reversion_model import MeanReversionModel
from reco_trading.core.momentum_model import MomentumModel
from reco_trading.infra.binance_client import BinanceClient


async def main() -> None:
    s = get_settings()
    client = BinanceClient(s.binance_api_key.get_secret_value(), s.binance_api_secret.get_secret_value(), s.binance_testnet)
    data = MarketDataService(client, s.symbol, s.timeframe)
    frame = FeatureEngine().build(await data.latest_ohlcv())

    MomentumModel().fit(frame)
    MeanReversionModel().fit(frame)
    await client.close()
    print('Modelos entrenados en memoria para validación operativa.')


if __name__ == '__main__':
    asyncio.run(main())
