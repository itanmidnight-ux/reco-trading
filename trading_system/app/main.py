from __future__ import annotations

import asyncio
import logging
import time

from fastapi import FastAPI
import uvicorn

from trading_system.app.config.settings import Settings
from trading_system.app.core.event_bus import Event, EventBus
from trading_system.app.core.rate_limiter import BinanceRateLimitController
from trading_system.app.database.repository import Repository
from trading_system.app.database.writer import AsyncDBWriter, WriteTask
from trading_system.app.models.ensemble.service import EnsembleService
from trading_system.app.services.decision_engine.service import DecisionEngineService
from trading_system.app.services.execution.service import ExecutionService, OrderRequest
from trading_system.app.services.feature_engineering.pipeline import FeatureEngineeringService
from trading_system.app.services.market_data.binance_client import BinanceClient
from trading_system.app.services.market_data.history_builder import OhlcvState
from trading_system.app.services.market_data.stream_handler import MarketStreamHandler
from trading_system.app.services.monitoring.service import MonitoringService
from trading_system.app.services.regime_detection.service import RegimeDetectionService
from trading_system.app.services.risk_management.service import RiskManagementService
from trading_system.app.services.sentiment.service import SentimentService

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)


class TradingSystem:
    def __init__(self) -> None:
        self.settings = Settings()
        self._enforce_startup_security()
        self.bus = EventBus()
        self.state = OhlcvState()

        self.rate_limiter = BinanceRateLimitController(self.settings.binance_max_weight, self.settings.order_rate_per_sec)
        self.binance = BinanceClient(self.settings, self.rate_limiter)
        self.stream = MarketStreamHandler(self.settings, self.bus, self.binance.ws_base)

        self.db = Repository(self.settings.postgres_dsn)
        self.db_writer = AsyncDBWriter()

        self.features = FeatureEngineeringService()
        self.regime = RegimeDetectionService()
        self.ensemble = EnsembleService()
        self.sentiment = SentimentService()
        self.decision = DecisionEngineService()
        self.risk = RiskManagementService(self.settings)
        self.execution = ExecutionService(self.settings, self.binance, on_executed=self.on_order_executed)
        self.monitoring = MonitoringService()
        self.monitoring.set_live_mode(self.settings.is_live_mode)

        self.last_status: dict[str, float | str] = {'status': 'booting'}
        self.last_market_event_ts = time.time()
        self.last_entry_price = 0.0

    def _enforce_startup_security(self) -> None:
        if self.settings.is_live_mode and not self.settings.enable_live_trading:
            raise RuntimeError('Startup blocked: live mode requested without enable_live_trading=true')

    async def bootstrap_history(self) -> None:
        klines = await self.binance.get_klines(self.settings.symbol, '1m', 500)
        for row in klines:
            self.state.high.append(float(row[2]))
            self.state.low.append(float(row[3]))
            self.state.close.append(float(row[4]))
            self.state.volume.append(float(row[5]))

    def register_handlers(self) -> None:
        symbol = self.settings.symbol.lower()
        self.bus.subscribe(f'market.{symbol}@kline_1m', self.on_kline)
        self.bus.subscribe(f'market.{symbol}@depth20@100ms', self.on_depth)

    async def on_order_executed(self, execution: dict) -> None:
        price = float(execution.get('price') or 0.0)
        side = str(execution.get('side', 'LONG'))
        status = str(execution.get('status', 'unknown'))
        pnl = 0.0
        if status in {'take_profit_filled', 'stop_loss_filled'} and self.last_entry_price and price:
            pnl_raw = (price - self.last_entry_price) / self.last_entry_price
            pnl = pnl_raw if side == 'LONG' else -pnl_raw
            self.last_entry_price = 0.0
        if status == 'entry_filled':
            self.last_entry_price = price or self.last_entry_price

        self.risk.update(pnl)
        self.monitoring.metrics.register_trade(pnl)
        logger.info(
            'Order lifecycle | symbol=%s side=%s status=%s qty=%.6f price=%.8f',
            execution.get('symbol'),
            side,
            status,
            float(execution.get('qty') or 0.0),
            price,
        )
        await self.db_writer.submit(
            WriteTask(
                fn=self.db.save,
                kwargs={
                    'table': 'order_executions',
                    'payload': {
                        'ts': int(time.time() * 1000),
                        'symbol': execution['symbol'],
                        'side': side,
                        'qty': float(execution['qty']),
                        'price': price,
                        'status': status,
                        'pnl': pnl,
                    },
                },
            )
        )

    async def on_depth(self, event: Event) -> None:
        self.last_market_event_ts = time.time()
        self.state.ingest_depth(event.payload)

    async def on_kline(self, event: Event) -> None:
        self.last_market_event_ts = time.time()
        self.state.ingest_kline(event.payload)
        if len(self.state.close) < 80:
            return

        fv = self.features.build(self.state)
        regime = self.regime.detect(fv)
        ensemble_out = self.ensemble.infer(fv, regime)
        sentiment = await self.sentiment.latest()
        decision = self.decision.decide(ensemble_out, sentiment, fv)
        self.monitoring.metrics.update_score(decision.score)

        price = self.state.close[-1]
        await self.db_writer.submit(
            WriteTask(
                fn=self.db.save,
                kwargs={
                    'table': 'candles',
                    'payload': {
                        'symbol': self.settings.symbol,
                        'interval': '1m',
                        'ts': int(time.time() * 1000),
                        'open': self.state.close[-2],
                        'high': self.state.high[-1],
                        'low': self.state.low[-1],
                        'close': price,
                        'volume': self.state.volume[-1],
                    },
                },
            )
        )
        await self.db_writer.submit(
            WriteTask(
                fn=self.db.save,
                kwargs={
                    'table': 'trade_signals',
                    'payload': {
                        'ts': int(time.time() * 1000),
                        'symbol': self.settings.symbol,
                        'signal': decision.signal,
                        'score': decision.score,
                        'expected_value': decision.expected_value,
                        'reason': decision.reason[:512],
                    },
                },
            )
        )

        plan = self.risk.plan(decision, fv, price)

        if plan.allow:
            await self.execution.submit(
                OrderRequest(
                    self.settings.symbol,
                    decision.signal,
                    plan.qty,
                    ref_price=price,
                    stop_loss=plan.stop_loss,
                    take_profit=plan.take_profit,
                    trailing_delta=plan.trailing,
                    trailing_activation=price,
                )
            )

        usage = await self.rate_limiter.usage_1m()
        drawdown = 1 - self.risk.equity / self.risk.peak
        ws_stale_seconds = time.time() - self.last_market_event_ts
        alerts = self.monitoring.alerts(usage, self.settings.binance_max_weight, drawdown, ws_stale_seconds)
        for alert in alerts:
            logger.warning('ALERTA: %s', alert)

        self.last_status = {
            'status': 'running',
            'signal': decision.signal,
            'score': round(decision.score, 4),
            'regime': regime.name,
            'weight_1m': usage,
            'ws_stale_seconds': round(ws_stale_seconds, 3),
            **self.monitoring.snapshot(),
        }
        logger.info('Decision %s | score=%.3f | regime=%s | EV=%.5f | %s', decision.signal, decision.score, regime.name, decision.expected_value, decision.reason)

    async def run(self) -> None:
        self.register_handlers()
        await self.db.initialize()
        asyncio.create_task(self.db_writer.run())
        await self.bootstrap_history()
        self.last_status = {'status': 'running'}
        await asyncio.gather(self.stream.run(), self.execution.run())


system = TradingSystem()
api = FastAPI(title='Trading System Gateway')


@api.get('/health')
async def health() -> dict:
    return {'ok': True, **system.last_status}


@api.get('/metrics/rate-limit')
async def rate_limit() -> dict:
    usage = await system.rate_limiter.usage_1m()
    return {'usage_1m': usage, 'limit': system.settings.binance_max_weight}


@api.get('/metrics/performance')
async def performance() -> dict:
    return system.monitoring.snapshot()


async def launch_services() -> None:
    await system.run()


if __name__ == '__main__':

    async def _main() -> None:
        task = asyncio.create_task(launch_services())
        config = uvicorn.Config(api, host='0.0.0.0', port=8000, log_level='info')
        server = uvicorn.Server(config)
        await asyncio.gather(task, server.serve())

    asyncio.run(_main())
