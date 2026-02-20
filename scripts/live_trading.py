from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from reco_trading.config.settings import get_settings
from reco_trading.core.execution_engine import ExecutionEngine
from reco_trading.core.feature_engine import FeatureEngine
from reco_trading.core.institutional_risk import InstitutionalRiskManager, RiskConfig
from reco_trading.core.market_data import MarketDataService
from reco_trading.core.market_regime import MarketRegimeDetector
from reco_trading.core.mean_reversion_model import MeanReversionModel
from reco_trading.core.momentum_model import MomentumModel
from reco_trading.core.pipeline import TradingPipeline
from reco_trading.core.signal_fusion_engine import SignalFusionEngine, SignalObservation
from reco_trading.infra.binance_client import BinanceClient
from reco_trading.infra.database import Database
from reco_trading.infra.state_manager import StateManager
from reco_trading.monitoring.health_check import HealthCheck
from reco_trading.core.portfolio_engine import PortfolioState


@dataclass
class RuntimeState:
    equity: float
    daily_pnl: float
    consecutive_losses: int
    last_signal: str = 'HOLD'


class MarketDataFeedAdapter:
    def __init__(self, market_data: MarketDataService, interval_seconds: int) -> None:
        self.market_data = market_data
        self.interval_seconds = interval_seconds

    async def stream(self):
        while True:
            yield await self.market_data.latest_ohlcv()
            await asyncio.sleep(self.interval_seconds)


class FeatureEngineAdapter:
    def __init__(
        self,
        feature_engine: FeatureEngine,
        momentum: MomentumModel,
        reversion: MeanReversionModel,
        state: RuntimeState,
    ):
        self.feature_engine = feature_engine
        self.momentum = momentum
        self.reversion = reversion
        self.state = state

    def compute(self, data: pd.DataFrame) -> dict:
        frame = self.feature_engine.build(data)
        last = frame.iloc[-1]

        momentum_up = self.momentum.predict_proba_up(frame)
        reversion_up = self.reversion.predict_reversion(frame)

        returns = frame['return'].tail(250).to_numpy(dtype=float)
        prices = frame['close'].tail(250)
        returns_df = pd.DataFrame({'BTCUSDT': frame['return'].tail(250).to_numpy(dtype=float)})

        confidence = abs(momentum_up - 0.5) * 2.0
        win_rate = float(np.clip(0.5 + confidence * 0.25, 0.5, 0.75))

        return {
            'returns': returns,
            'returns_df': returns_df,
            'prices': prices,
            'signals': {'momentum': momentum_up, 'mean_reversion': reversion_up},
            'volatility': float(last['volatility20']),
            'equity': float(max(self.state.equity, 1.0)),
            'atr': float(last['atr14']),
            'win_rate': win_rate,
            'reward_risk': 1.8,
        }


class FusionEngineAdapter:
    def __init__(self) -> None:
        self.engine = SignalFusionEngine(model_names=['momentum', 'mean_reversion'])

    def fuse(self, signals: dict, regime: str, volatility: float) -> float:
        observations = [
            SignalObservation(
                name='momentum',
                score=2.0 * float(signals['momentum']) - 1.0,
                confidence=abs(float(signals['momentum']) - 0.5) * 2.0,
                regime_weight=1.15 if regime == 'trend' else 0.95,
                volatility_adjustment=float(np.clip(1.0 - (volatility * 3.0), 0.5, 1.1)),
                historical_precision=0.56,
            ),
            SignalObservation(
                name='mean_reversion',
                score=2.0 * float(signals['mean_reversion']) - 1.0,
                confidence=abs(float(signals['mean_reversion']) - 0.5) * 2.0,
                regime_weight=1.10 if regime == 'range' else 0.9,
                volatility_adjustment=float(np.clip(1.0 - (volatility * 2.0), 0.6, 1.1)),
                historical_precision=0.54,
            ),
        ]
        return self.engine.fuse(observations).calibrated_probability


class ExecutionEngineAdapter:
    def __init__(self, execution: ExecutionEngine, state: RuntimeState) -> None:
        self.execution = execution
        self.state = state

    async def execute(self, side: str, size: float) -> None:
        fill = await self.execution.execute(side, size)
        if not fill:
            return

        self.state.last_signal = side
        logger.info(f"FILL confirmado id={fill.get('id')} side={fill.get('side')} px={fill.get('average')}")


class InstitutionalTradingPipeline:
    def __init__(self) -> None:
        self.s = get_settings()
        self.client = BinanceClient(
            self.s.binance_api_key.get_secret_value(),
            self.s.binance_api_secret.get_secret_value(),
            self.s.binance_testnet,
        )
        self.db = Database(self.s.postgres_dsn, self.s.postgres_admin_dsn)
        self.state_manager = StateManager(self.s.redis_url)
        loaded = self.state_manager.load()
        self.state = RuntimeState(
            equity=float(loaded.equity),
            daily_pnl=float(loaded.daily_pnl),
            consecutive_losses=int(loaded.consecutive_losses),
            last_signal=str(loaded.last_signal),
        )

        self.data = MarketDataService(self.client, self.s.symbol, self.s.timeframe)
        self.feature_engine = FeatureEngineAdapter(FeatureEngine(), MomentumModel(), MeanReversionModel(), self.state)
        self.regime = MarketRegimeDetector(n_states=3)
        self.fusion = FusionEngineAdapter()
        self.risk = InstitutionalRiskManager(
            RiskConfig(
                risk_per_trade=self.s.risk_per_trade,
                max_daily_loss=self.s.max_daily_loss,
                max_drawdown=self.s.max_global_drawdown,
                max_exposure=self.s.max_asset_exposure,
                max_correlation=self.s.correlation_threshold,
                kelly_fraction=0.5,
                max_consecutive_losses=self.s.max_consecutive_losses,
            )
        )
        self.risk.daily_pnl = self.state.daily_pnl
        self.risk.consecutive_losses = self.state.consecutive_losses
        self.risk.update_equity(self.state.equity)

        self.execution = ExecutionEngineAdapter(
            ExecutionEngine(self.client, self.s.symbol, self.db, redis_url=self.s.redis_url),
            self.state,
        )

        self.pipeline = TradingPipeline(
            data_feed=MarketDataFeedAdapter(self.data, self.s.loop_interval_seconds),
            feature_engine=self.feature_engine,
            regime_detector=self.regime,
            fusion_engine=self.fusion,
            risk_manager=self.risk,
            execution_engine=self.execution,
            queue_maxsize=512,
        )

    async def run(self) -> None:
        await self.db.init()
        health = await HealthCheck().run(self.client, self.s.symbol, self.s.timeframe)
        if not health['ok']:
            raise RuntimeError('Health check falló: no se pudo obtener OHLCV de Binance')

        logger.info('Pipeline async institucional (queue-driven) iniciado')
        try:
            await self.pipeline.run()
        finally:
            self.state_manager.save(
                PortfolioState(
                    equity=self.state.equity,
                    daily_pnl=self.risk.daily_pnl,
                    consecutive_losses=self.risk.consecutive_losses,
                    last_signal=self.state.last_signal,
                )
            )
            await self.client.close()
            await self.db.close()


async def main() -> None:
    await InstitutionalTradingPipeline().run()


if __name__ == '__main__':
    asyncio.run(main())
