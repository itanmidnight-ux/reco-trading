from __future__ import annotations

import asyncio
import importlib
import os
import resource
from dataclasses import dataclass
from loguru import logger


@dataclass
class RuntimeTuningReport:
    event_loop_policy: str
    uvloop_enabled: bool
    nofile_soft: int
    nofile_hard: int
    cpu_affinity: set[int] | None
    postgres_pool: dict[str, int]
    redis_pool: dict[str, int]


class RuntimeOptimizer:
    """Aplica ajustes de runtime seguros para producción."""

    def __init__(
        self,
        *,
        enable_uvloop: bool = True,
        min_nofile: int = 4096,
        target_nofile: int = 65_535,
        cpu_affinity: set[int] | None = None,
    ) -> None:
        self.enable_uvloop = enable_uvloop
        self.min_nofile = min_nofile
        self.target_nofile = target_nofile
        self.cpu_affinity = cpu_affinity

    def apply(self) -> RuntimeTuningReport:
        policy = self._configure_event_loop_policy()
        uvloop_enabled = self._enable_uvloop_if_available(policy)
        nofile_soft, nofile_hard = self._validate_and_apply_nofile()
        affinity = self._apply_cpu_affinity()
        postgres_pool, redis_pool = self.recommended_pools()

        return RuntimeTuningReport(
            event_loop_policy=policy,
            uvloop_enabled=uvloop_enabled,
            nofile_soft=nofile_soft,
            nofile_hard=nofile_hard,
            cpu_affinity=affinity,
            postgres_pool=postgres_pool,
            redis_pool=redis_pool,
        )

    def _configure_event_loop_policy(self) -> str:
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            policy = 'WindowsSelectorEventLoopPolicy'
        else:
            policy = asyncio.get_event_loop_policy().__class__.__name__

        logger.info('Runtime event loop policy configurada', policy=policy)
        return policy

    def _enable_uvloop_if_available(self, policy: str) -> bool:
        if os.name == 'nt' or not self.enable_uvloop:
            return False

        if 'uvloop' in policy.lower():
            return True

        try:
            uvloop = importlib.import_module('uvloop')
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            logger.info('uvloop habilitado')
            return True
        except ModuleNotFoundError:
            logger.info('uvloop no instalado, se mantiene loop estándar')
            return False
        except Exception as exc:
            logger.warning('No se pudo habilitar uvloop', error=str(exc))
            return False

    def _validate_and_apply_nofile(self) -> tuple[int, int]:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < self.min_nofile:
            desired_soft = min(max(self.target_nofile, self.min_nofile), hard)
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (desired_soft, hard))
                soft = desired_soft
                logger.info('RLIMIT_NOFILE ajustado', soft=soft, hard=hard)
            except (ValueError, PermissionError) as exc:
                logger.warning('No se pudo ajustar RLIMIT_NOFILE', soft=soft, hard=hard, error=str(exc))
        return soft, hard

    def _apply_cpu_affinity(self) -> set[int] | None:
        if not self.cpu_affinity:
            return None
        if not hasattr(os, 'sched_setaffinity'):
            logger.warning('Afinidad CPU no soportada en este SO')
            return None

        os.sched_setaffinity(0, self.cpu_affinity)
        active = set(os.sched_getaffinity(0))
        logger.info('Afinidad CPU aplicada', cpus=sorted(active))
        return active

    @staticmethod
    def recommended_pools() -> tuple[dict[str, int], dict[str, int]]:
        cpus = max((os.cpu_count() or 2), 2)
        postgres = {
            'min_size': 4,
            'max_size': min(8 * cpus, 128),
            'command_timeout_seconds': 15,
        }
        redis = {
            'max_connections': min(16 * cpus, 256),
            'health_check_interval_seconds': 30,
            'socket_timeout_seconds': 5,
        }
        logger.info('Pools recomendados calculados', postgres=postgres, redis=redis)
        return postgres, redis


__all__ = ['RuntimeOptimizer', 'RuntimeTuningReport']
