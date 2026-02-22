from __future__ import annotations

from enum import Enum


class SystemState(str, Enum):
    WAITING_FOR_DATA = 'WAITING_FOR_DATA'
    LEARNING_MARKET = 'LEARNING_MARKET'
    ANALYZING_MARKET = 'ANALYZING_MARKET'
    SENDING_ORDER = 'SENDING_ORDER'
    IN_POSITION = 'IN_POSITION'
    BLOCKED_BY_RISK = 'BLOCKED_BY_RISK'
    ERROR = 'ERROR'
