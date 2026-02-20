from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from loguru import logger

from reco_trading.core.event_pipeline import AsyncEventBus, PipelineEvent
from reco_trading.evolution.evolution_engine import EvolutionEngine


HealthSnapshotProvider = Callable[[], dict[str, Any]]


class EvolutionBackgroundService:
    """Servicio no bloqueante que evalúa/muta/despliega en segundo plano."""

    def __init__(
        self,
        *,
        engine: EvolutionEngine,
        event_bus: AsyncEventBus,
        health_snapshot_provider: HealthSnapshotProvider,
        interval_seconds: float = 60.0,
    ) -> None:
        self.engine = engine
        self.event_bus = event_bus
        self.health_snapshot_provider = health_snapshot_provider
        self.interval_seconds = max(interval_seconds, 1.0)

        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def _publish(self, topic: str, payload: dict[str, Any]) -> None:
        await self.event_bus.publish(PipelineEvent(topic=topic, payload=payload))

    async def _loop(self) -> None:
        logger.info('Evolution background service started')
        while not self._stop.is_set():
            try:
                snapshot = self.health_snapshot_provider()
                health = await self.engine.evaluate_system_health(**snapshot)
                await self._publish('evolution.health_evaluated', health)

                mutation = await self.engine.mutate_strategies(health)
                await self._publish('evolution.mutation_planned', mutation)

                deployment = await self.engine.deploy_new_configuration(mutation)
                await self._publish('evolution.configuration_deployed', deployment)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception('Evolution service cycle failed', exc=exc)
                await self._publish('evolution.error', {'error': str(exc)})

            await asyncio.sleep(self.interval_seconds)

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None
