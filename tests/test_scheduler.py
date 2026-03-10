from __future__ import annotations

import asyncio

from reco_trading.core.scheduler import EngineScheduler, ScheduledTask


async def _run_scheduler_once() -> int:
    scheduler = EngineScheduler()
    counter = {"value": 0}

    async def tick() -> None:
        counter["value"] += 1

    scheduler.register(ScheduledTask(name="tick", interval_seconds=0.05, coroutine=tick))
    await asyncio.sleep(0.18)
    await scheduler.shutdown()
    return counter["value"]


def test_scheduler_executes_tasks() -> None:
    calls = asyncio.run(_run_scheduler_once())
    assert calls >= 2
