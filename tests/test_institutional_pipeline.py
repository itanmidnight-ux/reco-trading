import asyncio

from reco_trading.core.event_pipeline import AsyncEventBus
from reco_trading.core.institutional_pipeline import (
    InstitutionalFeatureFlags,
    InstitutionalTradingPipeline,
    StrategyRoutingConfig,
)


class _Step:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def process(self, payload):
        self.calls.append(self.name)
        return {'stage': self.name, 'payload': payload}


def test_institutional_pipeline_full_chain_and_strategy_routing():
    calls = []

    strategy_calls = []

    class _Strategy:
        def __init__(self, name):
            self.name = name

        def handle(self, payload):
            strategy_calls.append((self.name, payload.get('selected_strategy')))

    async def _run():
        bus = AsyncEventBus()
        pipeline = InstitutionalTradingPipeline(
            bus,
            data_module=_Step('data', calls),
            microstructure_module=_Step('microstructure', calls),
            transformer_module=_Step('transformer', calls),
            regime_module=_Step('regime', calls),
            stacking_module=_Step('stacking', calls),
            meta_learning_module=_Step('meta_learning', calls),
            rl_module=_Step('rl', calls),
            risk_module=_Step('risk', calls),
            portfolio_module=_Step('portfolio', calls),
            strategy_handlers={
                'arbitrage': _Strategy('arbitrage'),
                'market_making': _Strategy('market_making'),
                'directional': _Strategy('directional'),
            },
            routing=StrategyRoutingConfig(priorities={'arbitrage': 90, 'market_making': 100, 'directional': 80}),
        )
        await pipeline.start(workers=1)
        await pipeline.submit({'strategy_scores': {'arbitrage': 1.0, 'market_making': 0.2, 'directional': 0.8}})
        await asyncio.wait_for(bus._queue.join(), timeout=1)
        await pipeline.shutdown()

    asyncio.run(_run())

    assert calls == ['data', 'microstructure', 'transformer', 'regime', 'stacking', 'meta_learning', 'rl', 'risk', 'portfolio']
    assert strategy_calls == [('market_making', 'market_making')]


def test_institutional_pipeline_feature_flag_and_fallback():
    strategy_calls = []
    fallback_calls = []

    class _Fallback:
        def handle(self, payload):
            fallback_calls.append(payload)

    class _Strategy:
        def handle(self, payload):
            strategy_calls.append(payload)

    async def _run():
        bus = AsyncEventBus()
        pipeline = InstitutionalTradingPipeline(
            bus,
            feature_flags=InstitutionalFeatureFlags(strategy_routing=False, meta_learning=False),
            meta_learning_module=lambda payload: (_ for _ in ()).throw(RuntimeError('disabled module should not run')),
            legacy_fallback=_Fallback(),
            strategy_handlers={'arbitrage': _Strategy()},
        )
        await pipeline.start(workers=1)
        await pipeline.submit({'foo': 'bar'})
        await asyncio.wait_for(bus._queue.join(), timeout=1)
        await pipeline.shutdown()

    asyncio.run(_run())

    assert len(fallback_calls) == 1
    assert strategy_calls == []


def test_institutional_pipeline_uses_transformer_inference_engine():
    calls = []

    class _InferenceEngine:
        async def infer(self, model_name, payload):
            calls.append((model_name, payload))
            return {'engine': model_name}

    async def _run():
        bus = AsyncEventBus()
        pipeline = InstitutionalTradingPipeline(
            bus,
            transformer_module=lambda payload: (_ for _ in ()).throw(RuntimeError('should not run direct transformer')),
            transformer_inference_engine=_InferenceEngine(),
            transformer_model_name='orderflow-v2',
        )
        await pipeline.start(workers=1)
        await pipeline.submit({'sequence_features': [1, 2, 3]})
        await asyncio.wait_for(bus._queue.join(), timeout=1)
        await pipeline.shutdown()

    asyncio.run(_run())

    assert calls == [('orderflow-v2', [1, 2, 3])]
