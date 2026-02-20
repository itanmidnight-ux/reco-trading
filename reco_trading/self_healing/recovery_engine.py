from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from loguru import logger


class ModuleRestarter(Protocol):
    def restart(self, module_name: str) -> None: ...


class StrategyController(Protocol):
    def disable_temporarily(self, strategy_name: str, ttl_seconds: int) -> None: ...


class ExchangeRouter(Protocol):
    def switch_to_backup(self, exchange: str, backup_exchange: str) -> None: ...


class ConfigManager(Protocol):
    def rollback_to_stable(self) -> dict[str, Any]: ...


@dataclass(slots=True)
class RecoveryAction:
    action: str
    status: str
    detail: str
    timestamp: float = field(default_factory=time.time)


class RecoveryEngine:
    def __init__(
        self,
        *,
        module_restarter: ModuleRestarter | None = None,
        strategy_controller: StrategyController | None = None,
        exchange_router: ExchangeRouter | None = None,
        config_manager: ConfigManager | None = None,
    ) -> None:
        self.module_restarter = module_restarter
        self.strategy_controller = strategy_controller
        self.exchange_router = exchange_router
        self.config_manager = config_manager

    def restart_module(self, module_name: str) -> RecoveryAction:
        if self.module_restarter is None:
            return RecoveryAction('restart_module', 'skipped', f'No hay ModuleRestarter para {module_name}')
        self.module_restarter.restart(module_name)
        logger.bind(component='self_healing', action='restart_module', module=module_name).warning('Reinicio de módulo ejecutado')
        return RecoveryAction('restart_module', 'ok', f'Módulo reiniciado: {module_name}')

    def disable_strategy_temporarily(self, strategy_name: str, *, ttl_seconds: int = 300) -> RecoveryAction:
        if self.strategy_controller is None:
            return RecoveryAction('disable_strategy', 'skipped', f'No hay StrategyController para {strategy_name}')
        self.strategy_controller.disable_temporarily(strategy_name, ttl_seconds)
        logger.bind(component='self_healing', action='disable_strategy', strategy=strategy_name).warning('Estrategia desactivada temporalmente')
        return RecoveryAction('disable_strategy', 'ok', f'Estrategia {strategy_name} desactivada por {ttl_seconds}s')

    def activate_conservative_fallback(self) -> RecoveryAction:
        if self.strategy_controller is None:
            return RecoveryAction('conservative_fallback', 'skipped', 'No hay StrategyController')
        self.strategy_controller.disable_temporarily('aggressive_mm', ttl_seconds=900)
        self.strategy_controller.disable_temporarily('cross_exchange_arbitrage', ttl_seconds=900)
        logger.bind(component='self_healing', action='conservative_fallback').critical('Fallback conservador activado')
        return RecoveryAction('conservative_fallback', 'ok', 'Estrategias agresivas deshabilitadas')

    def switch_to_backup_exchange(self, exchange: str, backup_exchange: str) -> RecoveryAction:
        if self.exchange_router is None:
            return RecoveryAction('switch_backup_exchange', 'skipped', f'Sin ExchangeRouter para {exchange}')
        self.exchange_router.switch_to_backup(exchange, backup_exchange)
        logger.bind(component='self_healing', action='switch_backup_exchange', exchange=exchange, backup=backup_exchange).critical('Switch a exchange backup ejecutado')
        return RecoveryAction('switch_backup_exchange', 'ok', f'Switch {exchange} -> {backup_exchange}')

    def rollback_to_stable_configuration(self) -> RecoveryAction:
        if self.config_manager is None:
            return RecoveryAction('rollback_configuration', 'skipped', 'Sin ConfigManager')
        snapshot = self.config_manager.rollback_to_stable()
        logger.bind(component='self_healing', action='rollback_configuration', snapshot=snapshot).critical('Rollback de configuración estable ejecutado')
        return RecoveryAction('rollback_configuration', 'ok', 'Configuración restaurada a snapshot estable')
