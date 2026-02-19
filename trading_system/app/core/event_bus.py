from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

Handler = Callable[['Event'], Awaitable[None]]


@dataclass(slots=True)
class Event:
    topic: str
    payload: dict[str, Any]
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Handler]] = defaultdict(list)

    def subscribe(self, topic: str, handler: Handler) -> None:
        self._subs[topic].append(handler)

    async def publish(self, event: Event) -> None:
        handlers = self._subs.get(event.topic, [])
        if handlers:
            await asyncio.gather(*(h(event) for h in handlers))
