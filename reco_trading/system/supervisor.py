from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from loguru import logger


@dataclass
class RestartPolicy:
    max_retries: int = 5
    backoff_base_seconds: float = 1.0
    backoff_max_seconds: float = 30.0

    def next_backoff(self, retries: int) -> float:
        return min(self.backoff_base_seconds * (2 ** max(0, retries - 1)), self.backoff_max_seconds)


@dataclass
class CriticalModule:
    name: str
    task_factory: Callable[[], Awaitable[None]]
    policy: RestartPolicy = field(default_factory=RestartPolicy)
    task: asyncio.Task[None] | None = None
    retries: int = 0
    next_restart_at: float = 0.0


class KernelSupervisor:
    def __init__(
        self,
        *,
        monitor_interval_seconds: float = 5.0,
        memory_window: int = 6,
        leak_growth_bytes: int = 150 * 1024 * 1024,
        on_fatal: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.monitor_interval_seconds = monitor_interval_seconds
        self.memory_window = memory_window
        self.leak_growth_bytes = leak_growth_bytes
        self.on_fatal = on_fatal
        self._modules: dict[str, CriticalModule] = {}
        self._stop_event = asyncio.Event()
        self._baseline_cpu = time.process_time()
        self._baseline_wall = time.monotonic()
        self._rss_samples: deque[int] = deque(maxlen=memory_window)

    def register_module(self, name: str, task_factory: Callable[[], Awaitable[None]], policy: RestartPolicy | None = None) -> None:
        self._modules[name] = CriticalModule(name=name, task_factory=task_factory, policy=policy or RestartPolicy())

    async def run(self) -> None:
        for module in self._modules.values():
            self._start_module(module)

        while not self._stop_event.is_set():
            await self._watchdog_modules()
            self._inspect_system_health()
            await asyncio.sleep(self.monitor_interval_seconds)

    async def stop(self) -> None:
        self._stop_event.set()
        tasks = [m.task for m in self._modules.values() if m.task is not None and not m.task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _start_module(self, module: CriticalModule) -> None:
        module.task = asyncio.create_task(module.task_factory(), name=f'supervised:{module.name}')
        logger.info('Supervisor inició módulo crítico', module=module.name)

    async def _watchdog_modules(self) -> None:
        now = time.monotonic()
        for module in self._modules.values():
            task = module.task
            if task is None:
                continue
            if not task.done():
                continue

            exc = task.exception()
            if exc is None:
                reason = f'módulo {module.name} finalizó inesperadamente sin excepción'
            else:
                reason = f'módulo {module.name} falló: {exc}'

            module.retries += 1
            if module.retries > module.policy.max_retries:
                logger.error('Supervisor agotó reintentos', module=module.name, retries=module.retries)
                if self.on_fatal is not None:
                    await self.on_fatal(reason)
                self._stop_event.set()
                return

            backoff = module.policy.next_backoff(module.retries)
            module.next_restart_at = now + backoff
            logger.warning('Supervisor programó reinicio de módulo', module=module.name, reason=reason, backoff=backoff)

            if now >= module.next_restart_at:
                self._start_module(module)
            else:
                await asyncio.sleep(max(module.next_restart_at - now, 0.0))
                self._start_module(module)

    def _inspect_system_health(self) -> None:
        zombies = self._detect_zombie_processes()
        if zombies:
            logger.warning('Procesos zombie detectados', zombies=zombies)

        cpu_pct = self._cpu_percent()
        rss = self._memory_rss_bytes()
        if rss > 0:
            self._rss_samples.append(rss)
        leak_detected = self._memory_leak_heuristic()

        logger.info(
            'Supervisor health sample',
            cpu_percent=round(cpu_pct, 2),
            rss_mb=round(rss / (1024 * 1024), 2),
            memory_leak_suspected=leak_detected,
        )

    def _detect_zombie_processes(self) -> list[int]:
        zombies: list[int] = []
        for status_file in Path('/proc').glob('[0-9]*/status'):
            try:
                content = status_file.read_text(encoding='utf-8')
            except OSError:
                continue
            if '\nState:\tZ' in content or '\nState:\tZ (zombie)' in content:
                try:
                    zombies.append(int(status_file.parts[-2]))
                except ValueError:
                    continue
        return zombies

    def _cpu_percent(self) -> float:
        now_cpu = time.process_time()
        now_wall = time.monotonic()
        elapsed_cpu = max(now_cpu - self._baseline_cpu, 0.0)
        elapsed_wall = max(now_wall - self._baseline_wall, 1e-6)
        self._baseline_cpu = now_cpu
        self._baseline_wall = now_wall
        cores = max(os.cpu_count() or 1, 1)
        return (elapsed_cpu / elapsed_wall) * 100.0 / cores

    @staticmethod
    def _memory_rss_bytes() -> int:
        try:
            with open('/proc/self/status', 'r', encoding='utf-8') as fh:
                for line in fh:
                    if line.startswith('VmRSS:'):
                        return int(line.split()[1]) * 1024
        except OSError:
            return 0
        return 0

    def _memory_leak_heuristic(self) -> bool:
        if len(self._rss_samples) < max(3, self.memory_window // 2):
            return False

        growth = self._rss_samples[-1] - self._rss_samples[0]
        monotonic = all(b >= a for a, b in zip(self._rss_samples, list(self._rss_samples)[1:]))
        return monotonic and growth >= self.leak_growth_bytes


__all__ = ['KernelSupervisor', 'RestartPolicy']
