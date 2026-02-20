from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from typing import Any


class TradingPipeline:
    def __init__(
        self,
        data_feed,
        feature_engine,
        regime_detector,
        fusion_engine,
        risk_manager,
        execution_engine,
        *,
        buy_threshold: float = 0.70,
        sell_threshold: float = 0.30,
        queue_maxsize: int = 256,
    ):
        self.data_queue = asyncio.Queue(maxsize=queue_maxsize)
        self.signal_queue = asyncio.Queue(maxsize=queue_maxsize)

        self.data_feed = data_feed
        self.feature_engine = feature_engine
        self.regime_detector = regime_detector
        self.fusion_engine = fusion_engine
        self.risk_manager = risk_manager
        self.execution_engine = execution_engine

        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task[Any]] = []
        self._error_handler: Callable[[Exception], None] | None = None

    def set_error_handler(self, handler: Callable[[Exception], None]) -> None:
        self._error_handler = handler

    async def produce_market_data(self):
        try:
            async for data in self.data_feed.stream():
                if self._stop.is_set():
                    break
                await self.data_queue.put(data)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._error_handler:
                self._error_handler(exc)
            raise

    async def process_features(self):
        while not self._stop.is_set():
            data = await self.data_queue.get()
            try:
                features = self.feature_engine.compute(data)
                await self.signal_queue.put(features)
            finally:
                self.data_queue.task_done()

    async def decision_loop(self):
        while not self._stop.is_set():
            features = await self.signal_queue.get()
            try:
                equity = float(features['equity'])
                self.risk_manager.update_equity(equity)
                self.risk_manager.check_kill_switch(equity)

                if self.risk_manager.kill_switch:
                    continue

                returns_df = features.get('returns_df')
                if returns_df is not None and self.risk_manager.check_correlation_risk(returns_df):
                    continue

                regime = self.regime_detector.predict(features['returns'], features['prices'])
                probability = self.fusion_engine.fuse(
                    features['signals'],
                    regime['regime'],
                    float(features['volatility']),
                )

                if probability > self.buy_threshold:
                    side = 'BUY'
                elif probability < self.sell_threshold:
                    side = 'SELL'
                else:
                    continue

                position_size = self.risk_manager.calculate_position_size(
                    equity=equity,
                    atr=float(features['atr']),
                    win_rate=float(features['win_rate']),
                    reward_risk=float(features['reward_risk']),
                )

                if position_size > 0 and not self.risk_manager.kill_switch:
                    await self.execution_engine.execute(side, position_size)
            finally:
                self.signal_queue.task_done()

    async def run(self):
        self._tasks = [
            asyncio.create_task(self.produce_market_data()),
            asyncio.create_task(self.process_features()),
            asyncio.create_task(self.decision_loop()),
        ]
        try:
            await asyncio.gather(*self._tasks)
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            with suppress(Exception):
                await asyncio.gather(*self._tasks, return_exceptions=True)
