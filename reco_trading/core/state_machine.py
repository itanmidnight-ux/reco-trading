from __future__ import annotations

from enum import Enum


class BotState(str, Enum):
    INITIALIZING = "initializing"
    CONNECTING_EXCHANGE = "connecting_exchange"
    SYNCING_SYMBOL = "syncing_symbol"
    SYNCING_RULES = "syncing_rules"
    WAITING_MARKET_DATA = "waiting_market_data"
    ANALYZING_MARKET = "analyzing_market"
    SIGNAL_GENERATED = "signal_generated"
    PLACING_ORDER = "placing_order"
    POSITION_OPEN = "position_open"
    COOLDOWN = "cooldown"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"

    # Compatibility aliases for existing engine references.
    STARTING = INITIALIZING
    WAITING_SIGNAL = WAITING_MARKET_DATA
    MONITORING_POSITION = POSITION_OPEN
    ORDER_FILLED = POSITION_OPEN
    CLOSING_POSITION = COOLDOWN
