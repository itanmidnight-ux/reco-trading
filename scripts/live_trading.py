import asyncio

from loguru import logger

from reco_trading.config.settings import get_settings
from reco_trading.core.execution_engine import ExecutionEngine
from reco_trading.core.feature_engine import FeatureEngine
from reco_trading.core.fusion_engine import FusionEngine
from reco_trading.core.market_data import MarketDataService
from reco_trading.core.mean_reversion_model import MeanReversionModel
from reco_trading.core.momentum_model import MomentumModel
from reco_trading.core.portfolio_engine import PortfolioEngine
from reco_trading.core.risk_engine import RiskEngine
from reco_trading.infra.binance_client import BinanceClient
from reco_trading.infra.database import Database
from reco_trading.infra.state_manager import StateManager
from reco_trading.monitoring.health_check import HealthCheck


async def _preview_stream(data: MarketDataService, symbol_rest: str) -> None:
    async for candle in data.live_preview(symbol_rest):
        logger.info(
            f"LIVE PREVIEW | close={candle['close']:.2f} high={candle['high']:.2f} low={candle['low']:.2f} vol={candle['volume']:.4f}"
        )


async def main() -> None:
    s = get_settings()
    client = BinanceClient(s.binance_api_key.get_secret_value(), s.binance_api_secret.get_secret_value(), s.binance_testnet)
    db = Database(s.postgres_dsn)
    await db.init()

    state_manager = StateManager(s.redis_url)
    portfolio = PortfolioEngine(state_manager.load())

    data = MarketDataService(client, s.symbol, s.timeframe)
    momentum = MomentumModel()
    reversion = MeanReversionModel()
    fusion = FusionEngine()
    risk = RiskEngine(
        s.risk_per_trade,
        s.max_daily_drawdown,
        s.max_consecutive_losses,
        s.atr_stop_multiplier,
        s.volatility_target,
        s.circuit_breaker_volatility,
    )
    execution = ExecutionEngine(client, s.symbol, db)

    health = await HealthCheck().run(client, s.symbol, s.timeframe)
    if not health['ok']:
        raise RuntimeError('Health check falló: no se pudo obtener OHLCV de Binance.')

    preview_task = asyncio.create_task(_preview_stream(data, s.symbol_rest))
    logger.info('Sistema de trading iniciado para BTC/USDT Spot 5m.')

    try:
        while True:
            frame = FeatureEngine().build(await data.latest_ohlcv())
            momentum_up = momentum.predict_proba_up(frame)
            reversion_prob = reversion.predict_reversion(frame)
            signal = fusion.decide(momentum_up, reversion_prob)

            last = frame.iloc[-1]
            decision = risk.evaluate(
                equity=max(portfolio.state.equity, 1.0),
                daily_pnl=portfolio.state.daily_pnl,
                consecutive_losses=portfolio.state.consecutive_losses,
                price=float(last['close']),
                atr=float(last['atr14']),
                annualized_vol=float(last['volatility20'] * (365 * 24 * 12) ** 0.5),
                side=signal,
            )

            if signal in {'BUY', 'SELL'} and decision.allowed:
                fill = await execution.execute_market_order(signal, decision.size_btc)
                if fill:
                    logger.info(f"FILL confirmado | id={fill.get('id')} side={fill.get('side')} px={fill.get('average')}")
            else:
                logger.info(f'No trade | signal={signal} reason={decision.reason}')

            portfolio.state.last_signal = signal
            state_manager.save(portfolio.state)
            await db.snapshot_portfolio(portfolio.state)
            await asyncio.sleep(s.loop_interval_seconds)
    finally:
        preview_task.cancel()
        await client.close()
        await db.close()


if __name__ == '__main__':
    asyncio.run(main())
