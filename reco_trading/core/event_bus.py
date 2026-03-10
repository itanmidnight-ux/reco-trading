from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from queue import Empty, Queue
import threading
from typing import Any


@dataclass(slots=True)
class Event:
    topic: str
    payload: dict[str, Any]


class EventBus:
    """Thread-safe one-way event bus (engine publishes, UI subscribes)."""

    def __init__(self, maxsize: int = 1000) -> None:
        self._queue: Queue[Event] = Queue(maxsize=maxsize)
        self._subscribers: dict[str, list[Callable[[Event], None]]] = defaultdict(list)
        self._lock = threading.RLock()

    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        event = Event(topic=topic, payload=payload)
        try:
            self._queue.put_nowait(event)
        except Exception:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            self._queue.put_nowait(event)

    def subscribe(self, topic: str, callback: Callable[[Event], None]) -> None:
        with self._lock:
            self._subscribers[topic].append(callback)

    def drain(self, max_events: int = 100) -> int:
        processed = 0
        for _ in range(max_events):
            try:
                event = self._queue.get_nowait()
            except Empty:
                break
            callbacks = list(self._subscribers.get(event.topic, [])) + list(self._subscribers.get("*", []))
            for callback in callbacks:
                callback(event)
            processed += 1
        return processed
