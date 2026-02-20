from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from reco_trading.distributed.models import Heartbeat, TaskEnvelope, TaskResult, WorkerRegistration


class ClusterCoordinator:
    def __init__(
        self,
        redis_client: Any,
        *,
        heartbeat_ttl_s: int = 15,
        key_prefix: str = 'reco:cluster',
        monitor_interval_s: float = 2.0,
    ) -> None:
        self.redis = redis_client
        self.heartbeat_ttl_s = heartbeat_ttl_s
        self.key_prefix = key_prefix
        self.monitor_interval_s = monitor_interval_s
        self._stop_event = asyncio.Event()
        self._monitor_task: asyncio.Task[None] | None = None

    async def startup(self) -> None:
        self._stop_event.clear()
        self._monitor_task = asyncio.create_task(self.monitor_workers())

    async def shutdown(self) -> None:
        self._stop_event.set()
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            await asyncio.gather(self._monitor_task, return_exceptions=True)
            self._monitor_task = None

    async def register_worker(self, registration: WorkerRegistration) -> None:
        worker_key = self._worker_key(registration.worker_id)
        await self.redis.hset(
            worker_key,
            mapping={
                'worker_id': registration.worker_id,
                'task_types': json.dumps(sorted(registration.task_types)),
                'max_concurrency': registration.max_concurrency,
                'load': 0,
                'status': 'active',
                'metadata': json.dumps(registration.metadata),
                'registered_at': registration.registered_at.isoformat(),
            },
        )
        await self.redis.sadd(self._workers_key(), registration.worker_id)
        await self._update_heartbeat(Heartbeat(worker_id=registration.worker_id, load=0))

    async def dispatch_task(self, envelope: TaskEnvelope) -> str:
        worker_id = await self._select_worker(envelope.task_type, envelope.affinity_key)

        task_data = asdict(envelope)
        task_data['created_at'] = envelope.created_at.isoformat()
        task_data['status'] = 'queued' if worker_id is None else 'assigned'
        task_data['assigned_worker'] = worker_id
        await self.redis.hset(self._task_key(envelope.task_id), mapping=self._encode_mapping(task_data))

        if worker_id is None:
            await self.redis.zadd(self._queued_tasks_key(), {envelope.task_id: float(envelope.priority)})
            return envelope.task_id

        await self._assign_task_to_worker(envelope.task_id, worker_id)
        return envelope.task_id

    async def submit_task_result(self, result: TaskResult) -> None:
        task_key = self._task_key(result.task_id)
        current = await self.redis.hgetall(task_key)
        worker_id = result.worker_id or self._decode(current.get('assigned_worker'))

        update = asdict(result)
        update['completed_at'] = result.completed_at.isoformat()
        await self.redis.hset(task_key, mapping=self._encode_mapping(update))

        if worker_id:
            await self.redis.srem(self._worker_tasks_key(worker_id), result.task_id)
            await self.redis.hincrby(self._worker_key(worker_id), 'load', -1)

    async def update_worker_heartbeat(self, heartbeat: Heartbeat) -> None:
        await self._update_heartbeat(heartbeat)

    async def monitor_workers(self) -> None:
        while not self._stop_event.is_set():
            worker_ids = await self.redis.smembers(self._workers_key())
            for raw_worker_id in worker_ids:
                worker_id = self._decode(raw_worker_id)
                if not worker_id:
                    continue
                exists = await self.redis.exists(self._heartbeat_key(worker_id))
                if exists:
                    continue
                await self._handle_dead_worker(worker_id)
            await self._drain_queued_tasks()
            await asyncio.sleep(self.monitor_interval_s)

    async def _handle_dead_worker(self, worker_id: str) -> None:
        await self.redis.srem(self._workers_key(), worker_id)
        await self.redis.hset(self._worker_key(worker_id), mapping={'status': 'stale'})

        assigned = await self.redis.smembers(self._worker_tasks_key(worker_id))
        for raw_task_id in assigned:
            task_id = self._decode(raw_task_id)
            if not task_id:
                continue
            await self.redis.srem(self._worker_tasks_key(worker_id), task_id)
            await self.redis.hset(
                self._task_key(task_id),
                mapping={'status': 'requeued', 'assigned_worker': ''},
            )
            await self.redis.zadd(self._queued_tasks_key(), {task_id: 0.0})

    async def _drain_queued_tasks(self) -> None:
        queued_task_ids = await self.redis.zrevrange(self._queued_tasks_key(), 0, 49)
        for raw_task_id in queued_task_ids:
            task_id = self._decode(raw_task_id)
            if not task_id:
                continue
            task_data = await self.redis.hgetall(self._task_key(task_id))
            task_type = self._decode(task_data.get('task_type'))
            if not task_type:
                continue
            affinity_key = self._decode(task_data.get('affinity_key')) or None
            worker_id = await self._select_worker(task_type, affinity_key)
            if worker_id is None:
                continue
            await self.redis.zrem(self._queued_tasks_key(), task_id)
            await self.redis.hset(self._task_key(task_id), mapping={'status': 'assigned', 'assigned_worker': worker_id})
            await self._assign_task_to_worker(task_id, worker_id)

    async def _assign_task_to_worker(self, task_id: str, worker_id: str) -> None:
        await self.redis.sadd(self._worker_tasks_key(worker_id), task_id)
        await self.redis.hincrby(self._worker_key(worker_id), 'load', 1)
        await self.redis.rpush(self._worker_queue_key(worker_id), task_id)

    async def _select_worker(self, task_type: str, affinity_key: str | None = None) -> str | None:
        worker_ids = await self.redis.smembers(self._workers_key())
        candidates: list[tuple[int, int, str]] = []

        for raw_worker_id in worker_ids:
            worker_id = self._decode(raw_worker_id)
            if not worker_id:
                continue

            worker = await self.redis.hgetall(self._worker_key(worker_id))
            if self._decode(worker.get('status')) != 'active':
                continue

            task_types = set(json.loads(self._decode(worker.get('task_types')) or '[]'))
            supports_type = 1 if task_type in task_types else 0
            load = int(self._decode(worker.get('load')) or 0)
            max_concurrency = int(self._decode(worker.get('max_concurrency')) or 1)
            if load >= max_concurrency:
                continue

            affinity_bonus = 0
            if affinity_key:
                sticky_worker = self._decode(await self.redis.hget(self._affinity_key(), affinity_key))
                if sticky_worker == worker_id:
                    affinity_bonus = 2
            candidates.append((supports_type + affinity_bonus, -load, worker_id))

        if not candidates:
            return None

        candidates.sort(reverse=True)
        selected = candidates[0][2]
        if affinity_key:
            await self.redis.hset(self._affinity_key(), mapping={affinity_key: selected})
        return selected

    async def _update_heartbeat(self, heartbeat: Heartbeat) -> None:
        now = heartbeat.timestamp.astimezone(timezone.utc).isoformat()
        await self.redis.set(self._heartbeat_key(heartbeat.worker_id), now, ex=self.heartbeat_ttl_s)
        await self.redis.hset(
            self._worker_key(heartbeat.worker_id),
            mapping={'last_seen': now, 'load': heartbeat.load, 'status': 'active'},
        )

    def _encode_mapping(self, mapping: dict[str, Any]) -> dict[str, str]:
        return {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in mapping.items()}

    def _decode(self, value: Any) -> str:
        if value is None:
            return ''
        if isinstance(value, bytes):
            return value.decode('utf-8')
        return str(value)

    def _workers_key(self) -> str:
        return f'{self.key_prefix}:workers'

    def _worker_key(self, worker_id: str) -> str:
        return f'{self.key_prefix}:workers:{worker_id}'

    def _worker_tasks_key(self, worker_id: str) -> str:
        return f'{self.key_prefix}:workers:{worker_id}:tasks'

    def _worker_queue_key(self, worker_id: str) -> str:
        return f'{self.key_prefix}:workers:{worker_id}:queue'

    def _task_key(self, task_id: str) -> str:
        return f'{self.key_prefix}:tasks:{task_id}'

    def _queued_tasks_key(self) -> str:
        return f'{self.key_prefix}:tasks:queued'

    def _heartbeat_key(self, worker_id: str) -> str:
        return f'{self.key_prefix}:heartbeats:{worker_id}'

    def _affinity_key(self) -> str:
        return f'{self.key_prefix}:affinity'
