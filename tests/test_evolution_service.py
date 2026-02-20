import asyncio

from reco_trading.core.event_pipeline import AsyncEventBus
from reco_trading.evolution.evolution_engine import EvolutionEngine
from reco_trading.self_healing.evolution_service import EvolutionBackgroundService


def test_evolution_engine_mutates_when_unhealthy():
    async def _run():
        engine = EvolutionEngine(min_health_score=0.8)
        health = await engine.evaluate_system_health(
            daily_pnl=-0.08,
            consecutive_losses=3,
            kill_switch=False,
            latency_ms=1500,
        )
        assert health['healthy'] is False

        mutation = await engine.mutate_strategies(health)
        assert mutation['mutated'] is True

        deploy = await engine.deploy_new_configuration(mutation)
        assert deploy['deployed'] is True
        assert deploy['profile'].startswith('adaptive_profile_')

    asyncio.run(_run())


def test_evolution_background_service_publishes_events():
    events = []

    async def _handler(event):
        events.append(event.topic)

    async def _run():
        bus = AsyncEventBus(maxsize=16)
        engine = EvolutionEngine(min_health_score=0.99)

        bus.subscribe('evolution.health_evaluated', _handler)
        bus.subscribe('evolution.mutation_planned', _handler)
        bus.subscribe('evolution.configuration_deployed', _handler)

        service = EvolutionBackgroundService(
            engine=engine,
            event_bus=bus,
            health_snapshot_provider=lambda: {
                'daily_pnl': -0.06,
                'consecutive_losses': 2,
                'kill_switch': True,
            },
            interval_seconds=1.0,
        )

        await bus.start(workers=1)
        await service.start()
        await asyncio.sleep(1.2)
        await service.stop()
        await bus.shutdown()

    asyncio.run(_run())
    assert 'evolution.health_evaluated' in events
    assert 'evolution.mutation_planned' in events
    assert 'evolution.configuration_deployed' in events
