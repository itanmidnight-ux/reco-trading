"""
Reco-Trading Constants
Global constants used throughout the application.
Based on FreqTrade constants structure.
"""

from typing import Any, Literal

DOCS_LINK = "https://reco-trading.docs.example.com"
DEFAULT_CONFIG = "config.json"
PROCESS_THROTTLE_SECS = 5
RETRY_TIMEOUT = 30
TIMEOUT_UNITS = ["minutes", "seconds"]
EXPORT_OPTIONS = ["none", "trades", "signals"]

DEFAULT_DB_URL = "sqlite:///reco_trading.db"
DEFAULT_DB_DRYRUN_URL = "sqlite:///reco_trading.dryrun.db"

UNLIMITED_STAKE_AMOUNT = "unlimited"
DEFAULT_AMOUNT_RESERVE_PERCENT = 5.0

REQUIRED_ORDERTIF = ["entry", "exit"]
REQUIRED_ORDERTYPES = ["entry", "exit", "stoploss", "stoploss_on_exchange"]
PRICING_SIDES = ["ask", "bid", "same", "other"]
ORDERTYPE_POSSIBILITIES = ["limit", "market"]
ORDERTIF_POSSIBILITIES = ["GTC", "FOK", "IOC", "PO", "gtc", "fok", "ioc", "po"]

HYPEROPT_LOSS_BUILTIN = [
    "OnlyProfitHyperOptLoss",
    "SharpeHyperOptLoss",
    "SharpeHyperOptLossDaily",
    "SortinoHyperOptLoss",
    "SortinoHyperOptLossDaily",
    "CalmarHyperOptLoss",
    "MaxDrawDownHyperOptLoss",
    "MaxDrawDownRelativeHyperOptLoss",
    "MaxDrawDownPerPairHyperOptLoss",
    "ProfitDrawDownHyperOptLoss",
    "MultiMetricHyperOptLoss",
    "ShortTradeDurHyperOptLoss",
]

HYPEROPT_BUILTIN_SPACES = [
    "buy",
    "sell",
    "enter",
    "exit",
    "roi",
    "stoploss",
    "trailing",
    "protection",
    "trades",
]

HYPEROPT_BUILTIN_SPACE_OPTIONS = ["default", "all"] + HYPEROPT_BUILTIN_SPACES

AVAILABLE_PAIRLISTS = [
    "StaticPairList",
    "VolumePairList",
    "PercentChangePairList",
    "AgeFilter",
    "VolatilityFilter",
    "PrecisionFilter",
    "PriceFilter",
    "RangeStabilityFilter",
    "SpreadFilter",
    "ShuffleFilter",
]

AVAILABLE_PROTECTIONS = [
    "CooldownPeriod",
    "LowProfitPairs",
    "MaxDrawdownProtection",
    "StoplossGuard",
]

AVAILABLE_DATAHANDLERS = ["json", "jsongz", "feather", "parquet"]
BACKTEST_BREAKDOWNS = ["day", "week", "month", "year", "weekday"]
BACKTEST_CACHE_AGE = ["none", "day", "week", "month"]
BACKTEST_CACHE_DEFAULT = "day"
DRY_RUN_WALLET = 1000.0

DATETIME_PRINT_FORMAT = "%Y-%m-%d %H:%M:%S"
MATH_CLOSE_PREC = 1e-14
DEFAULT_DATAFRAME_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
DEFAULT_TRADES_COLUMNS = ["timestamp", "id", "type", "side", "price", "amount", "cost"]

TRADING_MODES = ["spot", "margin", "futures"]
MARGIN_MODES = ["cross", "isolated", ""]

LAST_BT_RESULT_FN = ".last_result.json"
FTHYPT_FILEVERSION = "fthypt_fileversion"

USERPATH_HYPEROPTS = "hyperopts"
USERPATH_STRATEGIES = "strategies"
USERPATH_FREQAIMODELS = "freqaimodels"
USERPATH_CONFIGS = "configs"

TELEGRAM_SETTING_OPTIONS = ["on", "off", "silent"]
WEBHOOK_FORMAT_OPTIONS = ["form", "json", "raw"]

CUSTOM_TAG_MAX_LENGTH = 255
DL_DATA_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

ENV_VAR_PREFIX = "RECO_TRADE__"

CANCELED_EXCHANGE_STATES = ("cancelled", "canceled", "expired", "rejected")
NON_OPEN_EXCHANGE_STATES = (*CANCELED_EXCHANGE_STATES, "closed")

DECIMAL_PER_COIN_FALLBACK = 3
DECIMALS_PER_COIN = {
    "BTC": 8,
    "ETH": 5,
    "USDT": 4,
    "USDC": 4,
}

DUST_PER_COIN = {"BTC": 0.0001, "ETH": 0.01, "USDT": 1.0}

SUPPORTED_FIAT = [
    "USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "CNY", "INR", "KRW",
    "BTC", "ETH", "XRP", "LTC", "BCH", "BNB", "SOL", "ADA", "DOT", "AVAX"
]

MINIMAL_CONFIG = {
    "stake_currency": "USDT",
    "dry_run": True,
    "exchange": {
        "name": "binance",
        "key": "",
        "secret": "",
        "pair_whitelist": [],
    },
}

CANCEL_REASON = {
    "TIMEOUT": "cancelled due to timeout",
    "PARTIALLY_FILLED_KEEP_OPEN": "partially filled - keeping order open",
    "PARTIALLY_FILLED": "partially filled",
    "FULLY_CANCELLED": "fully cancelled",
    "ALL_CANCELLED": "cancelled (all unfilled and partially filled open orders cancelled)",
    "CANCELLED_ON_EXCHANGE": "cancelled on exchange",
    "FORCE_EXIT": "forcesold",
    "REPLACE": "cancelled to be replaced by new limit order",
    "REPLACE_FAILED": "failed to replace order, deleting Trade",
    "USER_CANCEL": "user requested order cancel",
}

LongShort = Literal["long", "short"]
EntryExit = Literal["entry", "exit"]
BuySell = Literal["buy", "sell"]
MakerTaker = Literal["maker", "taker"]
BidAsk = Literal["bid", "ask"]

Config = dict[str, Any]
ExchangeConfig = dict[str, Any]
IntOrInf = float

EntryExecuteMode = Literal["initial", "pos_adjust", "replace"]

PAIR_PREFIXES = ["1000", "1000000", "1M", "K"]

VERSION = "1.0.0"
APP_NAME = "reco-trading"
