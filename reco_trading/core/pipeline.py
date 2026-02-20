from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from typing import Any

import numpy as np


class QuantKernel:
    def __init__(
        self,
        data_feed,
        feature_engine,
        regime_detector,
        fusion_engine,
        risk_manager,
        execution_engine,
        *,
        rl_agent=None,
        meta_learner=None,
        buy_threshold: float = 0.70,
        sell_threshold: float = 0.30,
        spread_factor: float = 1.0,
        queue_maxsize: int = 256,
        rl_persist_every: int = 25,
    ):
        self.data_queue = asyncio.Queue(maxsize=queue_maxsize)
        self.signal_queue = asyncio.Queue(maxsize=queue_maxsize)

        self.data_feed = data_feed
        self.feature_engine = feature_engine
        self.regime_detector = regime_detector
        self.fusion_engine = fusion_engine
        self.risk_manager = risk_manager
        self.execution_engine = execution_engine

        self.rl_agent = rl_agent
        self.meta_learner = meta_learner

        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.spread_factor = spread_factor
        self.rl_persist_every = max(int(rl_persist_every), 1)

        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task[Any]] = []
        self._error_handler: Callable[[Exception], None] | None = None
        self._decision_cycles = 0

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

    def _apply_rl_action(self, features: dict[str, Any], regime: str) -> tuple[bool, float, float, float, float]:
        risk_per_trade = float(self.risk_manager.config.risk_per_trade)
        buy_threshold = float(self.buy_threshold)
        sell_threshold = float(self.sell_threshold)
        spread_factor = float(self.spread_factor)

        if not self.rl_agent:
            return False, risk_per_trade, buy_threshold, sell_threshold, spread_factor

        equity = float(features.get('equity', 1.0))
        drawdown = float(np.clip(self.risk_manager.calculate_drawdown(equity), 0.0, 1.0))
        state = {
            'regime': regime,
            'volatility': float(features.get('volatility', 0.0)),
            'win_rate': float(features.get('win_rate', 0.5)),
            'drawdown': drawdown,
            'sharpe': float(features.get('sharpe', 0.0)),
            'obi': float(features.get('obi', 0.0)),
            'spread': float(features.get('spread', 0.0)),
            'transformer_prob_up': float(features.get('transformer_prob_up', 0.5)),
        }
        action = self.rl_agent.select_action(state)
        if action.pause_trading:
            return True, risk_per_trade, buy_threshold, sell_threshold, spread_factor

        risk_per_trade = float(np.clip(risk_per_trade + action.risk_per_trade_delta, 0.001, 0.05))
        buy_threshold = float(np.clip(buy_threshold + action.buy_threshold_delta, 0.45, 0.95))
        sell_threshold = float(np.clip(sell_threshold + action.sell_threshold_delta, 0.05, 0.55))
        spread_factor = float(np.clip(spread_factor + action.spread_factor_delta, 0.20, 3.00))
        if buy_threshold <= sell_threshold:
            midpoint = (buy_threshold + sell_threshold) / 2.0
            buy_threshold = float(np.clip(midpoint + 0.05, 0.45, 0.95))
            sell_threshold = float(np.clip(midpoint - 0.05, 0.05, 0.55))
        return False, risk_per_trade, buy_threshold, sell_threshold, spread_factor

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
                regime_name = str(regime.get('regime', 'unknown'))

                pause, risk_per_trade, buy_threshold, sell_threshold, spread_factor = self._apply_rl_action(features, regime_name)
                if pause:
                    continue

                self.risk_manager.config.risk_per_trade = risk_per_trade
                self.buy_threshold = buy_threshold
                self.sell_threshold = sell_threshold
                self.spread_factor = spread_factor

                meta_weights = None
                meta_confidence = 1.0
                if self.meta_learner:
                    drawdown = float(np.clip(self.risk_manager.calculate_drawdown(equity), 0.0, 1.0))
                    meta_output = self.meta_learner.optimize(
                        regime=regime_name,
                        volatility=float(features['volatility']),
                        drawdown=drawdown,
                    )
                    meta_weights = meta_output.model_weights
                    meta_confidence = meta_output.confidence_score

                probability = self.fusion_engine.fuse(
                    features['signals'],
                    regime_name,
                    float(features['volatility']),
                    meta_weights=meta_weights,
                    meta_confidence=meta_confidence,
                )

                if probability > self.buy_threshold:
                    side = 'BUY'
                elif probability < self.sell_threshold:
                    side = 'SELL'
                else:
                    continue

                position_size = self.risk_manager.calculate_position_size(
                    equity=equity,
                    atr=float(features['atr']) * self.spread_factor,
                    win_rate=float(features['win_rate']),
                    reward_risk=float(features['reward_risk']),
                )

                if position_size > 0 and not self.risk_manager.kill_switch:
                    await self.execution_engine.execute(side, position_size)

                self._decision_cycles += 1
                if self.rl_agent and self._decision_cycles % self.rl_persist_every == 0:
                    self.rl_agent.save_state()
            finally:
                self.signal_queue.task_done()

    async def run(self):
        if self.rl_agent:
            self.rl_agent.load_state()

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
        if self.rl_agent:
            self.rl_agent.save_state()


class TradingPipeline(QuantKernel):
    """Alias de compatibilidad."""

