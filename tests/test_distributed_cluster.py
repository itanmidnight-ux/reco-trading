import asyncio
import time

from reco_trading.distributed.coordinator import ClusterCoordinator
from reco_trading.distributed.models import TaskEnvelope, WorkerRegistration


class FakeRedis:
    def __init__(self):
        self.hashes = {}
        self.sets = {}
        self.values = {}
        self.sorted = {}
        self.lists = {}
        self.expiries = {}

    def _expired(self, key):
        exp = self.expiries.get(key)
        if exp is not None and exp <= time.time():
            self.values.pop(key, None)
            self.expiries.pop(key, None)
            return True
        return False

    async def hset(self, key, mapping):
        self.hashes.setdefault(key, {})
        self.hashes[key].update(mapping)

    async def hget(self, key, field):
        return self.hashes.get(key, {}).get(field)

    async def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    async def sadd(self, key, *values):
        self.sets.setdefault(key, set()).update(values)

    async def srem(self, key, *values):
        current = self.sets.setdefault(key, set())
        for value in values:
            current.discard(value)

    async def smembers(self, key):
        return set(self.sets.get(key, set()))

    async def set(self, key, value, ex=None):
        self.values[key] = value
        if ex is not None:
            self.expiries[key] = time.time() + ex

    async def exists(self, key):
        if self._expired(key):
            return 0
        return 1 if key in self.values else 0

    async def hincrby(self, key, field, amount):
        self.hashes.setdefault(key, {})
        current = int(self.hashes[key].get(field, 0))
        current += int(amount)
        self.hashes[key][field] = current
        return current

    async def zadd(self, key, mapping):
        self.sorted.setdefault(key, {})
        self.sorted[key].update(mapping)

    async def zrevrange(self, key, start, stop):
        data = self.sorted.get(key, {})
        ranked = sorted(data.items(), key=lambda item: item[1], reverse=True)
        if stop == -1:
            stop = len(ranked) - 1
        return [item[0] for item in ranked[start : stop + 1]]

    async def zrem(self, key, value):
        self.sorted.setdefault(key, {}).pop(value, None)

    async def rpush(self, key, value):
        self.lists.setdefault(key, []).append(value)


def test_coordinator_balancing_affinity_and_heartbeat_requeue():
    async def _run():
        redis = FakeRedis()
        coordinator = ClusterCoordinator(redis, heartbeat_ttl_s=1, monitor_interval_s=0.05)

        await coordinator.register_worker(
            WorkerRegistration(worker_id='w1', task_types={'features', 'inference'}, max_concurrency=2)
        )
        await coordinator.register_worker(
            WorkerRegistration(worker_id='w2', task_types={'features'}, max_concurrency=2)
        )

        t1 = TaskEnvelope(task_type='features', payload={'data': [1]}, affinity_key='btc')
        t2 = TaskEnvelope(task_type='features', payload={'data': [2]}, affinity_key='btc')
        await coordinator.dispatch_task(t1)
        await coordinator.dispatch_task(t2)

        assigned1 = (await redis.hgetall(coordinator._task_key(t1.task_id))).get('assigned_worker')
        assigned2 = (await redis.hgetall(coordinator._task_key(t2.task_id))).get('assigned_worker')
        assert assigned1 == assigned2

        await asyncio.sleep(1.1)
        monitor_task = asyncio.create_task(coordinator.monitor_workers())
        await asyncio.sleep(0.2)
        coordinator._stop_event.set()
        await asyncio.gather(monitor_task, return_exceptions=True)

        stale_worker = assigned1
        status = (await redis.hgetall(coordinator._worker_key(stale_worker))).get('status')
        task_state = (await redis.hgetall(coordinator._task_key(t1.task_id))).get('status')
        assert status == 'stale'
        assert task_state in {'requeued', 'assigned'}

    asyncio.run(_run())
