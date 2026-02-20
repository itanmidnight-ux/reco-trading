from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

from reco_trading.ai.rl_agent import TradingRLAgent
from reco_trading.config.settings import get_settings
from reco_trading.core.event_pipeline import AsyncEventBus, PipelineEvent
from reco_trading.core.execution_engine import ExecutionEngine
from reco_trading.core.feature_engine import FeatureEngine
from reco_trading.core.institutional_risk import InstitutionalRiskManager, RiskConfig
from reco_trading.core.market_data import MarketDataService
from reco_trading.core.market_regime import MarketRegimeDetector
from reco_trading.core.mean_reversion_model import MeanReversionModel
from reco_trading.core.meta_learning import AdaptiveMetaLearner
from reco_trading.core.microstructure import OrderBookMicrostructureAnalyzer
from reco_trading.core.momentum_model import MomentumModel
from reco_trading.core.pipeline import TradingPipeline
from reco_trading.core.portfolio_engine import PortfolioState
from reco_trading.core.signal_fusion_engine import SignalFusionEngine, SignalObservation
from reco_trading.evolution import EvolutionEngine, ProbabilisticModelContract, ReversionModelContract
from reco_trading.infra.binance_client import BinanceClient
from reco_trading.infra.database import Database
from reco_trading.infra.state_manager import StateManager
from reco_trading.monitoring.alert_manager import AlertManager
from reco_trading.monitoring.health_check import HealthCheck
from reco_trading.monitoring.metrics import MetricsExporter, TradingMetrics
from reco_trading.research.metrics import aggregate_execution_quality
from reco_trading.self_healing import EvolutionBackgroundService
from reco_trading.system.runtime import RuntimeOptimizer
from reco_trading.system.supervisor import KernelSupervisor, RestartPolicy


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
        momentum: ProbabilisticModelContract,
        reversion: ReversionModelContract,
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
    def __init__(self, execution: ExecutionEngine, state: RuntimeState, metrics: TradingMetrics, alert_manager: AlertManager) -> None:
        self.execution = execution
        self.state = state
        self.metrics = metrics
        self.alert_manager = alert_manager
        self._fills: list[dict] = []
        self._quotes: list[dict] = []
        self._midprice_path: list[dict] = []
        self._cancel_events: list[dict] = []

    async def execute(self, side: str, size: float) -> None:
        self.metrics.observe_request('execution', exchange='binance', strategy='live')
        fill = await self.execution.execute(side, size)
        if not fill:
            self.metrics.observe_error('execution', 'empty_fill', exchange='binance', strategy='live')
            self.metrics.set_fill_ratio('binance', 0.0, exchange='binance', strategy='live')
            self.alert_manager.evaluate_slo_alerts(
                error_rate=0.01,
                p95_latency_seconds=0.010,
                fill_ratio=0.0,
                drawdown_ratio=0.0,
                capital_protection_active=False,
                exchange='binance',
            )
            return

        self.state.last_signal = side
        price = float(fill.get('average') or fill.get('price') or 0.0)
        quantity = float(fill.get('filled') or fill.get('amount') or size)
        timestamp = fill.get('datetime') or datetime.utcnow().isoformat()
        exchange = str(fill.get('exchange') or 'BINANCE')

        self._fills.append(
            {
                'timestamp': timestamp,
                'side': side,
                'price': price,
                'quantity': quantity,
                'exchange': exchange,
                'strategy': 'live',
            }
        )
        self._quotes.append(
            {
                'timestamp': timestamp,
                'bid': price,
                'ask': price,
                'bid_size': quantity,
                'ask_size': quantity,
                'exchange': exchange,
                'strategy': 'live',
            }
        )
        self._midprice_path.append(
            {
                'timestamp': timestamp,
                'midprice': price,
                'exchange': exchange,
                'strategy': 'live',
            }
        )

        report = aggregate_execution_quality(
            fills=self._fills,
            quotes=self._quotes,
            midprice_path=self._midprice_path,
            cancel_events=self._cancel_events,
        )

        fill_ratio = float(report.get('fill_ratio', 1.0))
        self.metrics.set_fill_ratio('binance', fill_ratio, exchange='binance', strategy='live')
        self.metrics.observe_fill_quality(
            float(report.get('fill_quality_ratio', 1.0)),
            float(report.get('slippage_bps', 0.0)),
            exchange='binance',
            strategy='live',
        )
        self.alert_manager.evaluate_slo_alerts(
            error_rate=0.0,
            p95_latency_seconds=0.010,
            fill_ratio=fill_ratio,
            drawdown_ratio=0.0,
            capital_protection_active=False,
            exchange='binance',
            extra_payload=report,
        )

        logger.info(f"FILL confirmado id={fill.get('id')} side={fill.get('side')} px={fill.get('average')}")
        logger.info('Execution quality live report', report=report)


class QuantKernel:
    def __init__(self) -> None:
        self.s = get_settings()
        self._initialized = False
        self._shutdown_event = asyncio.Event()
        self._shutdown_lock = asyncio.Lock()
        self._worker_tasks: list[asyncio.Task[None]] = []
        self._supervisor_task: asyncio.Task[None] | None = None
        self.supervisor = KernelSupervisor(
            monitor_interval_seconds=max(float(self.s.loop_interval_seconds), 2.0),
            on_fatal=self.emergency_shutdown,
        )

    async def initialize(self) -> None:
        if self._initialized:
            return

        affinity = self._parse_cpu_affinity(self.s.runtime_cpu_affinity)
        runtime_report = RuntimeOptimizer(
            enable_uvloop=self.s.runtime_enable_uvloop,
            min_nofile=self.s.runtime_min_nofile,
            target_nofile=self.s.runtime_target_nofile,
            cpu_affinity=affinity,
        ).apply()
        logger.info('Runtime optimizado', report=runtime_report)

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

        self.alert_manager = AlertManager()
        self.metrics = TradingMetrics()
        self.metrics_exporter = MetricsExporter(port=self.s.monitoring_metrics_port, addr=self.s.monitoring_metrics_host)

        self.execution = ExecutionEngineAdapter(
            ExecutionEngine(
                self.client,
                self.s.symbol,
                self.db,
                redis_url=self.s.redis_url,
                order_timeout_seconds=self.s.execution_order_timeout_seconds,
            ),
            self.state,
            self.metrics,
            self.alert_manager,
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
        self.event_bus = AsyncEventBus(maxsize=1024)
        self.evolution_service = EvolutionBackgroundService(
            engine=EvolutionEngine(),
            event_bus=self.event_bus,
            health_snapshot_provider=self._health_snapshot,
            interval_seconds=max(float(self.s.loop_interval_seconds) * 6.0, 30.0),
        )
        self.rl_agent = TradingRLAgent(redis_url=self.s.redis_url)
        self.meta_learner = AdaptiveMetaLearner(model_names=['momentum', 'mean_reversion'])
        self.microstructure = OrderBookMicrostructureAnalyzer()
        self._initialized = True

    def _health_snapshot(self) -> dict:
        drawdown_ratio = min(max(abs(float(self.risk.daily_pnl)) / max(float(self.state.equity), 1.0), 0.0), 1.0)
        self.metrics.set_drawdown('daily', drawdown_ratio, exchange='binance', strategy='live')
        self.metrics.set_worker_health('healthy', True, exchange='binance', strategy='live', worker_id='pipeline')
        return {
            'daily_pnl': float(self.risk.daily_pnl),
            'consecutive_losses': int(self.risk.consecutive_losses),
            'kill_switch': bool(self.risk.kill_switch),
        }

    @staticmethod
    async def _on_evolution_event(event: PipelineEvent) -> None:
        logger.info('Evolution event', topic=event.topic, payload=event.payload)

    @staticmethod
    def _parse_cpu_affinity(raw: str | None) -> set[int] | None:
        if not raw:
            return None
        cpus: set[int] = set()
        for item in raw.split(','):
            item = item.strip()
            if not item:
                continue
            if item.isdigit():
                cpus.add(int(item))
        return cpus or None

    async def run(self) -> None:
        if not self._initialized:
            await self.initialize()

        await self.db.init()
        if self.s.monitoring_metrics_enabled:
            self.metrics_exporter.start()
            logger.info(f"Prometheus metrics endpoint activo en http://{self.s.monitoring_metrics_host}:{self.s.monitoring_metrics_port}/metrics")

        self.event_bus.subscribe('evolution.health_evaluated', self._on_evolution_event)
        self.event_bus.subscribe('evolution.mutation_planned', self._on_evolution_event)
        self.event_bus.subscribe('evolution.configuration_deployed', self._on_evolution_event)
        self.event_bus.subscribe('evolution.error', self._on_evolution_event)
        await self.event_bus.start(workers=1)
        await self.evolution_service.start()

        health = await HealthCheck().run(self.client, self.s.symbol, self.s.timeframe)
        if not health['ok']:
            await self.emergency_shutdown('health-check bootstrap falló')
            raise RuntimeError('Health check falló: no se pudo obtener OHLCV de Binance')

        logger.info('QuantKernel iniciado: pipeline + control de capital + validación live + RL/meta-learning + ejecución')
        self.supervisor.register_module('signal_pipeline', self.pipeline.run, RestartPolicy(max_retries=8, backoff_base_seconds=1.0, backoff_max_seconds=20.0))
        self.supervisor.register_module('global_risk_check', self.global_risk_check, RestartPolicy(max_retries=5, backoff_base_seconds=1.0, backoff_max_seconds=15.0))
        self.supervisor.register_module('capital_allocation_cycle', self.capital_allocation_cycle, RestartPolicy(max_retries=5, backoff_base_seconds=1.0, backoff_max_seconds=15.0))
        self.supervisor.register_module('health_supervision', self.health_supervision, RestartPolicy(max_retries=5, backoff_base_seconds=1.0, backoff_max_seconds=20.0))

        self._supervisor_task = asyncio.create_task(self.supervisor.run(), name='kernel_supervisor')
        self._worker_tasks = [self._supervisor_task]

        try:
            done, _ = await asyncio.wait(self._worker_tasks, return_when=asyncio.FIRST_EXCEPTION)
            for task in done:
                error = task.exception()
                if error is not None:
                    raise error
            await self._shutdown_event.wait()
        except asyncio.CancelledError:
            await self.emergency_shutdown('kernel cancelado')
            raise
        except Exception as exc:
            await self.emergency_shutdown(f'error en worker: {exc}')
            raise
        else:
            await self.emergency_shutdown('shutdown coordinado')

    async def global_risk_check(self) -> None:
        interval = max(float(self.s.loop_interval_seconds), 1.0)
        while not self._shutdown_event.is_set():
            self.risk.check_kill_switch(float(self.state.equity))
            if self.risk.kill_switch:
                await self.emergency_shutdown('kill switch global activado por riesgo')
                return
            await asyncio.sleep(interval)

    async def capital_allocation_cycle(self) -> None:
        interval = max(float(self.s.loop_interval_seconds) * 2.0, 2.0)
        self.rl_agent.load_state()

        while not self._shutdown_event.is_set():
            drawdown = self.risk.calculate_drawdown(float(self.state.equity))
            meta = self.meta_learner.optimize(
                regime='live',
                volatility=0.01,
                drawdown=drawdown,
                base_weights={'momentum': 0.5, 'mean_reversion': 0.5},
            )
            action = self.rl_agent.select_action(
                {
                    'regime': 'live',
                    'volatility': 0.01,
                    'win_rate': 0.55,
                    'drawdown': drawdown,
                    'obi': 0.0,
                    'spread': 0.0,
                    'transformer_prob_up': 0.5,
                }
            )

            risk_per_trade = self.s.risk_per_trade * action.size_multiplier
            self.risk.config.risk_per_trade = float(np.clip(risk_per_trade, 0.001, 0.05))
            logger.info(
                'Capital allocation actualizado',
                risk_per_trade=self.risk.config.risk_per_trade,
                meta_confidence=meta.confidence_score,
                model_weights=meta.model_weights,
            )

            if action.pause_trading:
                self.risk.kill_switch = True
                await self.emergency_shutdown('RL pausó trading por política conservadora')
                return

            await asyncio.sleep(interval)

    async def health_supervision(self) -> None:
        interval = max(float(self.s.loop_interval_seconds) * 3.0, 3.0)
        checker = HealthCheck()

        while not self._shutdown_event.is_set():
            health = await checker.run(self.client, self.s.symbol, self.s.timeframe)
            if not health['ok']:
                await self.emergency_shutdown('validación live falló en health supervision')
                return

            book = await self.data.latest_order_book(limit=20)
            snapshot = self.microstructure.compute(book['bids'], book['asks'])
            self.metrics.set_worker_health('healthy', not snapshot.liquidity_shock, exchange='binance', strategy='live', worker_id='health')

            if snapshot.liquidity_shock:
                await self.emergency_shutdown('liquidity shock detectado en health supervision')
                return

            await asyncio.sleep(interval)

    async def emergency_shutdown(self, reason: str) -> None:
        async with self._shutdown_lock:
            if self._shutdown_event.is_set() and not self._worker_tasks:
                return

            logger.warning(f'Emergency shutdown: {reason}')
            self._shutdown_event.set()
            self.risk.kill_switch = True

            for task in self._worker_tasks:
                if not task.done() and task is not asyncio.current_task():
                    task.cancel()
            if self._worker_tasks:
                await asyncio.gather(*self._worker_tasks, return_exceptions=True)
                self._worker_tasks.clear()

            await self.supervisor.stop()
            if self._supervisor_task and not self._supervisor_task.done() and self._supervisor_task is not asyncio.current_task():
                self._supervisor_task.cancel()
                await asyncio.gather(self._supervisor_task, return_exceptions=True)
            self._supervisor_task = None

            await self.pipeline.shutdown()
            await self.evolution_service.stop()
            await self.event_bus.shutdown()
            self.rl_agent.save_state()

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
