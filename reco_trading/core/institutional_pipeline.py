from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from reco_trading.core.event_pipeline import AsyncEventBus, PipelineEvent


@dataclass(slots=True)
class InstitutionalFeatureFlags:
    microstructure: bool = True
    transformer: bool = True
    regime: bool = True
    stacking: bool = True
    meta_learning: bool = True
    reinforcement_learning: bool = True
    risk: bool = True
    portfolio: bool = True
    strategy_routing: bool = True


@dataclass(slots=True)
class StrategyRoutingConfig:
    priorities: dict[str, int] = field(
        default_factory=lambda: {
            'arbitrage': 100,
            'market_making': 80,
            'directional': 60,
        }
    )
    enabled: dict[str, bool] = field(
        default_factory=lambda: {
            'arbitrage': True,
            'market_making': True,
            'directional': True,
        }
    )


class InstitutionalTradingPipeline:
    TOPIC_DATA = 'pipeline.data'
    TOPIC_MICROSTRUCTURE = 'pipeline.microstructure'
    TOPIC_TRANSFORMER = 'pipeline.transformer'
    TOPIC_REGIME = 'pipeline.regime'
    TOPIC_STACKING = 'pipeline.stacking'
    TOPIC_META_LEARNING = 'pipeline.meta_learning'
    TOPIC_REINFORCEMENT_LEARNING = 'pipeline.reinforcement_learning'
    TOPIC_RISK = 'pipeline.risk'
    TOPIC_PORTFOLIO = 'pipeline.portfolio'
    TOPIC_STRATEGY = 'pipeline.strategy'
    STRATEGY_TOPICS = {
        'arbitrage': 'pipeline.strategy.arbitrage',
        'market_making': 'pipeline.strategy.market_making',
        'directional': 'pipeline.strategy.directional',
    }

    def __init__(
        self,
        event_bus: AsyncEventBus,
        *,
        data_module: Any | None = None,
        microstructure_module: Any | None = None,
        transformer_module: Any | None = None,
        regime_module: Any | None = None,
        stacking_module: Any | None = None,
        meta_learning_module: Any | None = None,
        rl_module: Any | None = None,
        risk_module: Any | None = None,
        portfolio_module: Any | None = None,
        strategy_handlers: dict[str, Any] | None = None,
        legacy_fallback: Any | None = None,
        feature_flags: InstitutionalFeatureFlags | None = None,
        routing: StrategyRoutingConfig | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.data_module = data_module
        self.microstructure_module = microstructure_module
        self.transformer_module = transformer_module
        self.regime_module = regime_module
        self.stacking_module = stacking_module
        self.meta_learning_module = meta_learning_module
        self.rl_module = rl_module
        self.risk_module = risk_module
        self.portfolio_module = portfolio_module
        self.strategy_handlers = strategy_handlers or {}
        self.legacy_fallback = legacy_fallback
        self.feature_flags = feature_flags or InstitutionalFeatureFlags()
        self.routing = routing or StrategyRoutingConfig()

    def register_handlers(self) -> None:
        self.event_bus.subscribe(self.TOPIC_DATA, self._handle_data)
        self.event_bus.subscribe(self.TOPIC_MICROSTRUCTURE, self._handle_microstructure)
        self.event_bus.subscribe(self.TOPIC_TRANSFORMER, self._handle_transformer)
        self.event_bus.subscribe(self.TOPIC_REGIME, self._handle_regime)
        self.event_bus.subscribe(self.TOPIC_STACKING, self._handle_stacking)
        self.event_bus.subscribe(self.TOPIC_META_LEARNING, self._handle_meta_learning)
        self.event_bus.subscribe(self.TOPIC_REINFORCEMENT_LEARNING, self._handle_reinforcement_learning)
        self.event_bus.subscribe(self.TOPIC_RISK, self._handle_risk)
        self.event_bus.subscribe(self.TOPIC_PORTFOLIO, self._handle_portfolio)
        self.event_bus.subscribe(self.TOPIC_STRATEGY, self._handle_strategy_routing)

        for strategy_name, topic in self.STRATEGY_TOPICS.items():
            if handler := self.strategy_handlers.get(strategy_name):
                self.event_bus.subscribe(topic, self._wrap_strategy_handler(handler))

    async def start(self, workers: int = 2) -> None:
        self.register_handlers()
        await self.event_bus.start(workers=workers)

    async def shutdown(self) -> None:
        await self.event_bus.shutdown()

    async def submit(self, data: dict[str, Any]) -> None:
        payload = await self._run_component(self.data_module, data)
        await self.event_bus.publish(PipelineEvent(self.TOPIC_DATA, {'data': payload}))

    async def _handle_data(self, event: PipelineEvent) -> None:
        await self.event_bus.publish(PipelineEvent(self.TOPIC_MICROSTRUCTURE, dict(event.payload)))

    async def _handle_microstructure(self, event: PipelineEvent) -> None:
        payload = dict(event.payload)
        if self.feature_flags.microstructure and self.microstructure_module is not None:
            payload['microstructure'] = await self._run_component(self.microstructure_module, payload)
        await self.event_bus.publish(PipelineEvent(self.TOPIC_TRANSFORMER, payload))

    async def _handle_transformer(self, event: PipelineEvent) -> None:
        payload = dict(event.payload)
        if self.feature_flags.transformer and self.transformer_module is not None:
            payload['transformer'] = await self._run_component(self.transformer_module, payload)
        await self.event_bus.publish(PipelineEvent(self.TOPIC_REGIME, payload))

    async def _handle_regime(self, event: PipelineEvent) -> None:
        payload = dict(event.payload)
        if self.feature_flags.regime and self.regime_module is not None:
            payload['regime'] = await self._run_component(self.regime_module, payload)
        await self.event_bus.publish(PipelineEvent(self.TOPIC_STACKING, payload))

    async def _handle_stacking(self, event: PipelineEvent) -> None:
        payload = dict(event.payload)
        if self.feature_flags.stacking and self.stacking_module is not None:
            payload['stacking'] = await self._run_component(self.stacking_module, payload)
        await self.event_bus.publish(PipelineEvent(self.TOPIC_META_LEARNING, payload))

    async def _handle_meta_learning(self, event: PipelineEvent) -> None:
        payload = dict(event.payload)
        if self.feature_flags.meta_learning and self.meta_learning_module is not None:
            payload['meta_learning'] = await self._run_component(self.meta_learning_module, payload)
        await self.event_bus.publish(PipelineEvent(self.TOPIC_REINFORCEMENT_LEARNING, payload))

    async def _handle_reinforcement_learning(self, event: PipelineEvent) -> None:
        payload = dict(event.payload)
        if self.feature_flags.reinforcement_learning and self.rl_module is not None:
            payload['rl'] = await self._run_component(self.rl_module, payload)
        await self.event_bus.publish(PipelineEvent(self.TOPIC_RISK, payload))

    async def _handle_risk(self, event: PipelineEvent) -> None:
        payload = dict(event.payload)
        if self.feature_flags.risk and self.risk_module is not None:
            payload['risk'] = await self._run_component(self.risk_module, payload)
        await self.event_bus.publish(PipelineEvent(self.TOPIC_PORTFOLIO, payload))

    async def _handle_portfolio(self, event: PipelineEvent) -> None:
        payload = dict(event.payload)
        if self.feature_flags.portfolio and self.portfolio_module is not None:
            payload['portfolio'] = await self._run_component(self.portfolio_module, payload)
        await self.event_bus.publish(PipelineEvent(self.TOPIC_STRATEGY, payload))

    async def _handle_strategy_routing(self, event: PipelineEvent) -> None:
        payload = dict(event.payload)
        if not self.feature_flags.strategy_routing:
            await self._run_legacy_fallback(payload)
            return

        selected = self._select_strategy(payload)
        if selected is None:
            await self._run_legacy_fallback(payload)
            return

        await self.event_bus.publish(
            PipelineEvent(self.STRATEGY_TOPICS[selected], {**payload, 'selected_strategy': selected})
        )

    def _select_strategy(self, payload: dict[str, Any]) -> str | None:
        scores = payload.get('strategy_scores') or {}

        ranked: list[tuple[int, float, str]] = []
        for strategy, priority in self.routing.priorities.items():
            if not self.routing.enabled.get(strategy, True):
                continue
            score = float(scores.get(strategy, 0.0))
            ranked.append((int(priority), score, strategy))

        if not ranked:
            return None

        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best_priority, best_score, best_strategy = ranked[0]
        if best_priority <= 0 and best_score <= 0.0:
            return None
        return best_strategy

    def _wrap_strategy_handler(self, handler: Any):
        async def _wrapped(event: PipelineEvent) -> None:
            await self._run_component(handler, event.payload)

        return _wrapped

    async def _run_legacy_fallback(self, payload: dict[str, Any]) -> None:
        if self.legacy_fallback is None:
            return
        await self._run_component(self.legacy_fallback, payload)

    async def _run_component(self, component: Any, payload: dict[str, Any]) -> Any:
        if component is None:
            return payload

        if callable(component):
            result = component(payload)
            if hasattr(result, '__await__'):
                return await result
            return result

        for method_name in ('process', 'compute', 'predict', 'transform', 'run', 'execute', 'handle'):
            method = getattr(component, method_name, None)
            if method is None:
                continue
            result = method(payload)
            if hasattr(result, '__await__'):
                return await result
            return result

        return payload
