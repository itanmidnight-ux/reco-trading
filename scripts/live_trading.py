from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
from loguru import logger

from reco_trading.ai.rl_agent import TradingRLAgent
from reco_trading.config.settings import get_settings
from reco_trading.core.event_pipeline import AsyncEventBus, PipelineEvent
from reco_trading.core.execution_engine import ExecutionEngine
from reco_trading.core.feature_engine import FeatureEngine
from reco_trading.core.institutional_risk_manager import InstitutionalRiskManager, RiskLimits
from reco_trading.core.market_data import MarketDataService
from reco_trading.core.market_regime_detector import MarketRegimeDetector
from reco_trading.core.mean_reversion_model import MeanReversionModel
from reco_trading.core.meta_learning import AdaptiveMetaLearner
from reco_trading.core.microstructure import MicrostructureSnapshot, OrderBookMicrostructureAnalyzer
from reco_trading.core.momentum_model import MomentumModel
from reco_trading.core.portfolio_engine import PortfolioEngine
from reco_trading.core.portfolio_exposure_manager import PortfolioExposureManager
from reco_trading.core.portfolio_optimization import ConvexPortfolioOptimizer
from reco_trading.core.signal_fusion_engine import SignalFusionEngine, SignalObservation
from reco_trading.core.volatility_guard import ExtremeVolatilityFilter
from reco_trading.infra.binance_client import BinanceClient
from reco_trading.infra.database import Database
from reco_trading.infra.state_manager import StateManager
from reco_trading.monitoring.health_check import HealthCheck


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
        self.portfolio = PortfolioEngine(self.state_manager.load())

        self.data = MarketDataService(self.client, self.s.symbol, self.s.timeframe)
        self.feature_engine = FeatureEngine()
        self.micro = OrderBookMicrostructureAnalyzer(depth_levels=10)
        self.momentum = MomentumModel()
        self.reversion = MeanReversionModel()
        self.regime = MarketRegimeDetector()
        self.meta = AdaptiveMetaLearner(model_names=["momentum", "mean_reversion"])
        self.fusion = SignalFusionEngine(model_names=["momentum", "mean_reversion"])
        self.rl_agent = TradingRLAgent(redis_url=self.s.redis_url)
        self.rl_agent.load_state()
        self.vol_filter = ExtremeVolatilityFilter()
        self.exposure = PortfolioExposureManager()
        self.optimizer = ConvexPortfolioOptimizer()

        limits = RiskLimits(
            risk_per_trade=self.s.risk_per_trade,
            max_daily_loss=self.s.max_daily_loss,
            max_global_drawdown=self.s.max_global_drawdown,
            max_total_exposure=self.s.max_total_exposure,
            max_asset_exposure=self.s.max_asset_exposure,
            correlation_threshold=self.s.correlation_threshold,
        )
        self.risk = InstitutionalRiskManager(limits=limits, atr_multiplier=self.s.atr_stop_multiplier)
        self.execution = ExecutionEngine(self.client, self.s.symbol, self.db)

        self.bus = AsyncEventBus(maxsize=128)
        self.last_frame: pd.DataFrame | None = None
        self._loop_lock = asyncio.Lock()

    async def _build_microstructure(self) -> MicrostructureSnapshot:
        book = await asyncio.wait_for(self.data.latest_order_book(limit=20), timeout=8)
        return self.micro.compute(book["bids"], book["asks"])

    async def _on_market_data(self, event: PipelineEvent) -> None:
        async with self._loop_lock:
            ohlcv = event.payload["ohlcv"]
            micro = event.payload["micro"]
            frame = self.feature_engine.build(ohlcv, microstructure=micro)
            self.last_frame = frame
            await self.bus.publish(PipelineEvent(topic="features", payload={"frame": frame, "micro": micro}))

    async def _on_features(self, event: PipelineEvent) -> None:
        frame = event.payload["frame"]
        micro: MicrostructureSnapshot = event.payload["micro"]
        regime_state = self.regime.detect(frame)
        vol_decision = self.vol_filter.evaluate(frame)
        momentum_up = self.momentum.predict_proba_up(frame)
        mean_reversion = self.reversion.predict_reversion(frame)

        drawdown = self.risk.current_drawdown(max(self.portfolio.state.equity, 1.0))
        last_vol = float(frame.iloc[-1]["volatility20"])
        meta = self.meta.optimize(regime=regime_state.regime, volatility=last_vol, drawdown=drawdown)

        regime_weight = 1.2 if regime_state.regime == "trend" else 0.9
        observations = [
            SignalObservation(
                name="momentum",
                score=2.0 * momentum_up - 1.0,
                confidence=abs(momentum_up - 0.5) * 2.0,
                regime_weight=regime_weight,
                volatility_adjustment=vol_decision.exposure_multiplier,
                historical_precision=0.55,
            ),
            SignalObservation(
                name="mean_reversion",
                score=2.0 * mean_reversion - 1.0,
                confidence=abs(mean_reversion - 0.5) * 2.0,
                regime_weight=1.1 if regime_state.regime == "range" else 0.85,
                volatility_adjustment=vol_decision.exposure_multiplier,
                historical_precision=0.53,
            ),
        ]
        fusion_result = self.fusion.fuse(
            observations,
            meta_weights=meta.model_weights,
            meta_confidence=meta.confidence_score,
        )
        await self.bus.publish(
            PipelineEvent(
                topic="signal",
                payload={
                    "frame": frame,
                    "regime": regime_state,
                    "vol_decision": vol_decision,
                    "fusion": fusion_result,
                    "micro": micro,
                },
            )
        )

    async def _on_signal(self, event: PipelineEvent) -> None:
        frame: pd.DataFrame = event.payload["frame"]
        regime_state = event.payload["regime"]
        vol_decision = event.payload["vol_decision"]
        fusion = event.payload["fusion"]
        micro: MicrostructureSnapshot = event.payload["micro"]

        if not vol_decision.allow_trading or micro.liquidity_shock:
            logger.warning("Trading suspendido por volatilidad extrema o shock de liquidez")
            return

        p_up = fusion.calibrated_probability
        last = frame.iloc[-1]
        annualized_vol = float(last["volatility20"] * np.sqrt(365 * 24 * 12))
        drawdown = self.risk.current_drawdown(max(self.portfolio.state.equity, 1.0))

        rl_action = self.rl_agent.select_action(
            {
                "volatility": annualized_vol,
                "regime": regime_state.regime,
                "win_rate": max(p_up, 1.0 - p_up),
                "drawdown": drawdown,
                "sharpe": 0.0,
                "obi": micro.obi,
                "spread": micro.spread,
            }
        )
        if rl_action.pause_trading:
            logger.warning("RL agent activó pausa táctica de trading")
            return

        p_buy = 0.57 + rl_action.threshold_shift
        p_sell = 0.43 - rl_action.threshold_shift
        side = "BUY" if p_up >= p_buy else "SELL" if p_up <= p_sell else "HOLD"
        if side == "HOLD":
            logger.info(f"No trade: prob={p_up:.4f} regime={regime_state.regime}")
            return

        returns_matrix = pd.DataFrame({self.s.symbol_rest: frame["return"].tail(250).to_numpy()})
        exposure_snapshot = self.exposure.evaluate(returns_matrix, {self.s.symbol_rest: self.portfolio.state.equity})
        allocation = self.optimizer.kelly_constrained(returns_matrix, max_drawdown=self.s.max_global_drawdown)
        allocation_weight = allocation.weights.get(self.s.symbol_rest, 1.0)
        adaptive_risk_per_trade = self.s.risk_per_trade + rl_action.risk_shift

        assessment = self.risk.assess(
            symbol=self.s.symbol_rest,
            side=side,
            equity=max(self.portfolio.state.equity, 1.0),
            daily_pnl=self.portfolio.state.daily_pnl,
            current_price=float(last["close"]),
            atr=float(last["atr14"]),
            annualized_volatility=annualized_vol,
            volatility_multiplier=vol_decision.exposure_multiplier,
            expected_win_rate=max(p_up, 1.0 - p_up),
            avg_win=0.012,
            avg_loss=0.008,
            returns_matrix=returns_matrix,
            microstructure=micro,
            risk_per_trade_override=adaptive_risk_per_trade,
        )

        adjusted_size = assessment.position_size * rl_action.size_multiplier * allocation_weight
        logger.info(
            f"signal={side} p={p_up:.4f} regime={regime_state.regime} conf={regime_state.confidence:.3f} "
            f"risk_allowed={assessment.allowed} size={adjusted_size:.6f} vpin={micro.vpin:.3f} "
            f"concentration={exposure_snapshot.concentration_risk:.3f}"
        )

        if not assessment.allowed:
            return

        fill = await self.execution.execute_market_order(
            side,
            adjusted_size,
            microstructure=micro,
            timeout_seconds=45,
        )
        if fill:
            logger.info(f"FILL confirmado id={fill.get('id')} side={fill.get('side')} px={fill.get('average')}")
            self.rl_agent.update_policy(
                {
                    "volatility": annualized_vol,
                    "regime": regime_state.regime,
                    "win_rate": max(p_up, 1.0 - p_up),
                    "drawdown": drawdown,
                    "sharpe": 0.0,
                    "obi": micro.obi,
                    "spread": micro.spread,
                },
                delta_equity=float(fill.get("cost", 0.0)) / max(self.portfolio.state.equity, 1.0),
                drawdown=drawdown,
            )
            est_pnl = float(fill.get("cost", 0.0)) * (1.0 if side == "BUY" else -1.0) * 0.0001
            self.risk.register_trade_result(est_pnl)

    async def run(self) -> None:
        await self.db.init()
        health = await HealthCheck().run(self.client, self.s.symbol, self.s.timeframe)
        if not health["ok"]:
            raise RuntimeError("Health check falló: no se pudo obtener OHLCV de Binance")

        self.bus.subscribe("market_data", self._on_market_data)
        self.bus.subscribe("features", self._on_features)
        self.bus.subscribe("signal", self._on_signal)
        await self.bus.start(workers=3)

        logger.info("Pipeline async institucional iniciado")
        try:
            while True:
                try:
                    ohlcv = await asyncio.wait_for(self.data.latest_ohlcv(), timeout=12)
                    micro = await self._build_microstructure()
                    await self.bus.publish(PipelineEvent(topic="market_data", payload={"ohlcv": ohlcv, "micro": micro}))
                except Exception as exc:
                    logger.exception(f"Error de loop de datos; fallback activado: {exc}")
                await asyncio.sleep(self.s.loop_interval_seconds)
        finally:
            self.rl_agent.save_state()
            self.state_manager.save(self.portfolio.state)
            await self.bus.shutdown()
            await self.client.close()
            await self.db.close()


async def main() -> None:
    await InstitutionalTradingPipeline().run()


if __name__ == '__main__':
    asyncio.run(main())
