from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from reco_trading.core.capital_governor import CapitalGovernor


@dataclass(slots=True)
class ConservativeMode:
    active: bool = False
    buy_threshold: float = 0.7
    sell_threshold: float = 0.3
    position_size_multiplier: float = 1.0
    max_strategies: int = 8


@dataclass(slots=True)
class KernelNotification:
    source: str
    severity: str
    detail: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class QuantKernel:
    """Núcleo de control cuantitativo para reaccionar a eventos de validación live."""

    def __init__(self, capital_governor: CapitalGovernor, *, strategy_limit: int = 8) -> None:
        self.capital_governor = capital_governor
        self.mode = ConservativeMode(max_strategies=max(strategy_limit, 1))
        self.notifications: list[KernelNotification] = []

    def notify_validation_event(self, severity: str, detail: str, payload: dict[str, Any]) -> KernelNotification:
        event = KernelNotification(source='live_validator', severity=severity, detail=detail, payload=payload)
        self.notifications.append(event)
        return event

    def reduce_capital_for_validation(self, severity: str) -> float:
        multiplier = {
            'normal': 1.0,
            'degradado': 0.70,
            'critico': 0.35,
        }.get(severity, 0.85)
        return self.capital_governor.reduce_capital(multiplier)

    def activate_conservative_mode(self, severity: str) -> ConservativeMode:
        if severity == 'normal':
            self.mode = ConservativeMode(max_strategies=self.mode.max_strategies)
            return self.mode

        self.mode.active = True
        if severity == 'degradado':
            self.mode.buy_threshold = 0.78
            self.mode.sell_threshold = 0.22
            self.mode.position_size_multiplier = 0.75
            self.mode.max_strategies = max(1, self.mode.max_strategies - 2)
        else:
            self.mode.buy_threshold = 0.85
            self.mode.sell_threshold = 0.15
            self.mode.position_size_multiplier = 0.50
            self.mode.max_strategies = max(1, self.mode.max_strategies - 4)
        return self.mode
