from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class WorkerRegistration:
    worker_id: str
    task_types: set[str]
    max_concurrency: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=_utc_now)


@dataclass(slots=True)
class TaskEnvelope:
    task_type: str
    payload: dict[str, Any]
    task_id: str = field(default_factory=lambda: str(uuid4()))
    priority: int = 0
    affinity_key: str | None = None
    created_at: datetime = field(default_factory=_utc_now)
    assigned_worker: str | None = None


@dataclass(slots=True)
class TaskResult:
    task_id: str
    worker_id: str
    status: str
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    completed_at: datetime = field(default_factory=_utc_now)


@dataclass(slots=True)
class Heartbeat:
    worker_id: str
    load: int
    timestamp: datetime = field(default_factory=_utc_now)
