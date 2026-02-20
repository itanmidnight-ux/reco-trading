from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from loguru import logger


@dataclass(slots=True)
class DefensiveModeState:
    active: bool = False
    exposure_multiplier: float = 1.0
    market_making_paused: bool = False
    arbitrage_disabled: bool = False
    activated_at: float | None = None
    reason: str = ''


@dataclass(slots=True)
class DefensiveAction:
    action: str
    status: str
    detail: str
    state: DefensiveModeState
    timestamp: float = field(default_factory=time.time)


class CapitalProtectionEngine:
    def __init__(self, *, severe_exposure_multiplier: float = 0.2) -> None:
        self.severe_exposure_multiplier = min(1.0, max(0.01, severe_exposure_multiplier))
        self.state = DefensiveModeState()

    def activate_severe_defensive_mode(self, reason: str) -> DefensiveAction:
        self.state.active = True
        self.state.exposure_multiplier = self.severe_exposure_multiplier
        self.state.market_making_paused = True
        self.state.arbitrage_disabled = True
        self.state.activated_at = time.time()
        self.state.reason = reason
        logger.bind(component='self_healing', action='activate_defensive_mode', reason=reason).critical(
            'Modo defensivo severo activado'
        )
        return DefensiveAction(
            action='activate_defensive_mode',
            status='ok',
            detail='Exposición reducida, MM pausado y arbitrage desactivado',
            state=self.snapshot(),
        )

    def deactivate_defensive_mode(self) -> DefensiveAction:
        self.state = DefensiveModeState()
        logger.bind(component='self_healing', action='deactivate_defensive_mode').warning('Modo defensivo desactivado')
        return DefensiveAction(
            action='deactivate_defensive_mode',
            status='ok',
            detail='Operación normal restaurada',
            state=self.snapshot(),
        )

    def snapshot(self) -> DefensiveModeState:
        return DefensiveModeState(**asdict(self.state))
