from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from reco_trading.distributed.coordinator import ClusterCoordinator
from reco_trading.distributed.models import Heartbeat, TaskResult, WorkerRegistration


TaskHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class DistributedWorker:
    def __init__(
        self,
        worker_id: str,
        coordinator: ClusterCoordinator,
        *,
        max_concurrency: int = 2,
        reconnect_delay_s: float = 2.0,
    ) -> None:
        self.worker_id = worker_id
        self.coordinator = coordinator
        self.max_concurrency = max_concurrency
        self.reconnect_delay_s = reconnect_delay_s
        self._stop_event = asyncio.Event()
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._consume_task: asyncio.Task[None] | None = None
        self._handlers: dict[str, TaskHandler] = {
            'features': self._compute_features,
            'inference': self._run_inference,
            'backtesting': self._run_backtest,
            'optimization': self._run_optimization,
        }

    async def startup(self) -> None:
        self._stop_event.clear()
        await self.coordinator.register_worker(
            WorkerRegistration(
                worker_id=self.worker_id,
                task_types=set(self._handlers),
                max_concurrency=self.max_concurrency,
            )
        )
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._consume_task = asyncio.create_task(self._consume_loop())

    async def shutdown(self) -> None:
        self._stop_event.set()
        for task in (self._heartbeat_task, self._consume_task):
            if task is None:
                continue
            task.cancel()
        await asyncio.gather(*[t for t in (self._heartbeat_task, self._consume_task) if t is not None], return_exceptions=True)
        self._heartbeat_task = None
        self._consume_task = None

    async def _heartbeat_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                queue_depth = await self.coordinator.redis.llen(self.coordinator._worker_queue_key(self.worker_id))
                await self.coordinator.update_worker_heartbeat(Heartbeat(worker_id=self.worker_id, load=queue_depth))
                await asyncio.sleep(max(self.coordinator.heartbeat_ttl_s / 3, 1.0))
            except Exception:
                await asyncio.sleep(self.reconnect_delay_s)
                await self._attempt_reconnect()

    async def _consume_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = await self.coordinator.redis.blpop(
                    self.coordinator._worker_queue_key(self.worker_id),
                    timeout=max(int(self.coordinator.monitor_interval_s), 1),
                )
                if not item:
                    continue
                _, raw_task_id = item
                task_id = self.coordinator._decode(raw_task_id)
                await self._execute_task(task_id)
            except Exception:
                await asyncio.sleep(self.reconnect_delay_s)
                await self._attempt_reconnect()

    async def _attempt_reconnect(self) -> None:
        if self._stop_event.is_set():
            return
        await self.coordinator.register_worker(
            WorkerRegistration(
                worker_id=self.worker_id,
                task_types=set(self._handlers),
                max_concurrency=self.max_concurrency,
                metadata={'reconnected_at': datetime.now(timezone.utc).isoformat()},
            )
        )

    async def _execute_task(self, task_id: str) -> None:
        task_data = await self.coordinator.redis.hgetall(self.coordinator._task_key(task_id))
        task_type = self.coordinator._decode(task_data.get('task_type'))
        payload_raw = self.coordinator._decode(task_data.get('payload')) or '{}'
        payload = self._parse_payload(payload_raw)
        handler = self._handlers.get(task_type)

        if handler is None:
            await self.coordinator.submit_task_result(
                TaskResult(task_id=task_id, worker_id=self.worker_id, status='failed', error=f'unsupported task_type: {task_type}')
            )
            return

        try:
            output = await handler(payload)
            status = 'completed'
            error = None
        except Exception as exc:
            output = {}
            status = 'failed'
            error = str(exc)

        await self.coordinator.submit_task_result(
            TaskResult(task_id=task_id, worker_id=self.worker_id, status=status, output=output, error=error)
        )

    def _parse_payload(self, payload_raw: str) -> dict[str, Any]:
        import json

        try:
            parsed = json.loads(payload_raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
        return {}

    async def _compute_features(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = payload.get('data', payload)
        return {
            'feature_count': len(data) if hasattr(data, '__len__') else 1,
            'features_ready': True,
        }

    async def _run_inference(self, payload: dict[str, Any]) -> dict[str, Any]:
        confidence = float(payload.get('confidence_hint', 0.5))
        signal = 'buy' if confidence >= 0.5 else 'sell'
        return {'signal': signal, 'confidence': confidence}

    async def _run_backtest(self, payload: dict[str, Any]) -> dict[str, Any]:
        returns = payload.get('returns', [0.0])
        pnl = sum(float(x) for x in returns)
        return {'pnl': pnl, 'num_trades': len(returns)}

    async def _run_optimization(self, payload: dict[str, Any]) -> dict[str, Any]:
        candidates = payload.get('candidates', [])
        if not candidates:
            return {'selected': None, 'score': 0.0}
        best = max(candidates, key=lambda item: item.get('score', 0.0))
        return {'selected': best.get('name'), 'score': float(best.get('score', 0.0))}
