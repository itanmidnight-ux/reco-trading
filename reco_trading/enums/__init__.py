"""
Enums for Reco-Trading
Bot state and trading mode enums.
"""

from enum import Enum


class State(str, Enum):
    """Bot states."""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    RELOAD_CONFIG = "reload"


class TradingMode(str, Enum):
    """Trading modes."""
    SPOT = "spot"
    MARGIN = "margin"
    FUTURES = "futures"


class RunMode(str, Enum):
    """Run modes."""
    LIVE = "live"
    DRY_RUN = "dry_run"
    BACKTEST = "backtest"
    HYPEROPT = "hyperopt"
    EDGE = "edge"
    WEBSERVER = "webserver"


class SignalType(str, Enum):
    """Signal types."""
    ENTRY = "entry"
    EXIT = "exit"


class SignalDirection(str, Enum):
    """Signal directions."""
    LONG = "long"
    SHORT = "short"


class ExitType(str, Enum):
    """Exit reasons."""
    EXIT_signal = "exit_signal"
    ROI = "roi"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP_LOSS = "trailing_stop_loss"
    EMERGENCY_EXIT = "emergency_exit"
    FORCE_EXIT = "force_exit"
    USER_EXIT = "user_exit"


class OrderType(str, Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LOSS_LIMIT = "stop_loss_limit"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"


class OrderSide(str, Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


class CandleType(str, Enum):
    """Candle types."""
    SPOT = "spot"
    FUTURES = "futures"
    MARK = "mark"
    INDEX = "index"
    PREMIUM_INDEX = "premium_index"


class RPCMessageType(str, Enum):
    """RPC message types."""
    STATUS = "status"
    WARNING = "warning"
    STARTUP = "startup"
    ENTRY = "entry"
    ENTRY_FILL = "entry_fill"
    ENTRY_CANCEL = "entry_cancel"
    EXIT = "exit"
    EXIT_FILL = "exit_fill"
    EXIT_CANCEL = "exit_cancel"
    PROTECTION_TRIGGER = "protection_trigger"
    PROTECTION_TRIGGER_GLOBAL = "protection_trigger_global"


class PriceType(str, Enum):
    """Price types for orders."""
    LAST = "last"
    MARK = "mark"
    INDEX = "index"
    PREMIUM_INDEX = "premium_index"


class MarginMode(str, Enum):
    """Margin modes."""
    CROSS = "cross"
    ISOLATED = "isolated"


__all__ = [
    "State",
    "TradingMode",
    "RunMode",
    "SignalType",
    "SignalDirection",
    "ExitType",
    "OrderType",
    "OrderSide",
    "CandleType",
    "RPCMessageType",
    "PriceType",
    "MarginMode",
]
