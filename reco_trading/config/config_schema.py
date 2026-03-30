"""
Configuration Schema Validation
Pydantic-based configuration validation for Reco-Trading.
Based on FreqTrade's config_schema.py
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from reco_trading.constants import (
    AVAILABLE_DATAHANDLERS,
    AVAILABLE_PAIRLISTS,
    AVAILABLE_PROTECTIONS,
    BACKTEST_BREAKDOWNS,
    BACKTEST_CACHE_AGE,
    DRY_RUN_WALLET,
    EXPORT_OPTIONS,
    HYPEROPT_LOSS_BUILTIN,
    MARGIN_MODES,
    ORDERTIF_POSSIBILITIES,
    ORDERTYPE_POSSIBILITIES,
    PRICING_SIDES,
    REQUIRED_ORDERTIF,
    SUPPORTED_FIAT,
    TELEGRAM_SETTING_OPTIONS,
    TIMEOUT_UNITS,
    TRADING_MODES,
    UNLIMITED_STAKE_AMOUNT,
    WEBHOOK_FORMAT_OPTIONS,
)


class ExchangeConfig(BaseModel):
    name: str = Field(description="Exchange name (e.g., binance)")
    key: str = Field(default="", description="API Key")
    secret: str = Field(default="", description="API Secret")
    password: str = Field(default="", description="API Password (optional)")
    pair_whitelist: list[str] = Field(default_factory=list, description="Trading pairs")
    pair_blacklist: list[str] = Field(default_factory=list, description="Excluded pairs")
    ccxt_config: dict = Field(default_factory=dict, description="CCXT additional config")
    ccxt_async_config: dict = Field(default_factory=dict, description="CCXT async config")
    log_responses: bool = Field(default=False, description="Log exchange responses")


class UnfilledTimeout(BaseModel):
    entry: int = Field(default=10, description="Entry order timeout in minutes")
    exit: int = Field(default=10, description="Exit order timeout in minutes")
    exit_timeout_count: int = Field(default=0, description="Exit timeout count")
    unit: Literal["minutes", "seconds"] = Field(default="minutes", description="Time unit")


class OrderTypes(BaseModel):
    entry: Literal["limit", "market"] = Field(default="limit")
    exit: Literal["limit", "market"] = Field(default="limit")
    emergency_exit: Literal["limit", "market"] = Field(default="market")
    force_exit: Literal["limit", "market"] = Field(default="market")
    force_entry: Literal["limit", "market"] = Field(default="market")
    stoploss: Literal["limit", "market"] = Field(default="market")
    stoploss_on_exchange: bool = Field(default=False)


class OrderTimeInForce(BaseModel):
    entry: Literal["GTC", "FOK", "IOC", "PO"] = Field(default="GTC")
    exit: Literal["GTC", "FOK", "IOC", "PO"] = Field(default="GTC")


class EntryPricing(BaseModel):
    price_side: Literal["ask", "bid", "same", "other"] = Field(default="same")
    use_order_book: bool = Field(default=True)
    order_book_top: int = Field(default=1, ge=1, le=10)
    price_last_balance: float = Field(default=0.0)
    check_depth_of_market: dict = Field(default_factory=lambda: {"enabled": False, "bids_to_ask_delta": 1})


class ExitPricing(BaseModel):
    price_side: Literal["ask", "bid", "same", "other"] = Field(default="same")
    use_order_book: bool = Field(default=True)
    order_book_top: int = Field(default=1, ge=1, le=10)
    price_last_balance: float = Field(default=0.0)


class TelegramConfig(BaseModel):
    enabled: bool = Field(default=False)
    token: str = Field(default="")
    chat_id: str = Field(default="")
    notification_settings: dict = Field(default_factory=dict)
    reload: bool = Field(default=True)
    balance_dust_level: float = Field(default=0.01)


class APIServerConfig(BaseModel):
    enabled: bool = Field(default=False)
    listen_ip_address: str = Field(default="127.0.0.1")
    listen_port: int = Field(default=8080, ge=1024, le=65535)
    verbosity: Literal["error", "warning", "info", "debug"] = Field(default="error")
    enable_openapi: bool = Field(default=False)
    jwt_secret_key: str = Field(default="")
    CORS_origins: list[str] = Field(default_factory=list)
    username: str = Field(default="reco_trader")
    password: str = Field(default="")
    ws_token: str = Field(default="")


class WebhookConfig(BaseModel):
    enabled: bool = Field(default=False)
    url: str = Field(default="")
    format: Literal["form", "json", "raw"] = Field(default="form")


class PairlistConfig(BaseModel):
    method: str = Field(description="Pairlist method name")
    config: dict = Field(default_factory=dict)


class ProtectionConfig(BaseModel):
    method: str = Field(description="Protection method name")
    config: dict = Field(default_factory=dict)


class BotConfig(BaseModel):
    max_open_trades: int = Field(default=3, ge=-1, description="Max open trades (-1 for unlimited)")
    stake_currency: str = Field(default="USDT", description="Stake currency")
    stake_amount: float | Literal["unlimited"] = Field(default=0.05, ge=0.0001, description="Stake amount per trade")
    tradable_balance_ratio: float = Field(default=0.99, ge=0.0, le=1.0, description="Tradable balance ratio")
    available_capital: float = Field(default=1000.0, ge=0, description="Total capital available")
    fiat_display_currency: str = Field(default="USD", description="Fiat display currency")
    dry_run: bool = Field(default=True, description="Dry run mode")
    dry_run_wallet: float = Field(default=DRY_RUN_WALLET, description="Dry run wallet balance")
    cancel_open_orders_on_exit: bool = Field(default=False)
    process_only_new_candles: bool = Field(default=False)
    timeframe: str = Field(default="5m", description="Trading timeframe")

    minimal_roi: dict[str, float] = Field(default_factory=dict, description="Minimal ROI table")
    stoploss: float = Field(default=-0.10, description="Stoploss value (as ratio)")
    
    trailing_stop: bool = Field(default=False)
    trailing_stop_positive: float | None = Field(default=None)
    trailing_stop_positive_offset: float = Field(default=0.0)
    trailing_only_offset_is_reached: bool = Field(default=False)
    
    use_exit_signal: bool = Field(default=True)
    exit_profit_only: bool = Field(default=False)
    exit_profit_offset: float = Field(default=0.0)
    ignore_roi_if_entry_signal: bool = Field(default=False)
    ignore_buying_expired_candle_after: int = Field(default=300)
    
    trading_mode: Literal["spot", "margin", "futures"] = Field(default="spot")
    margin_mode: Literal["cross", "isolated", ""] = Field(default="")
    
    unfilledtimeout: UnfilledTimeout = Field(default_factory=UnfilledTimeout)
    order_types: OrderTypes = Field(default_factory=OrderTypes)
    order_time_in_force: OrderTimeInForce = Field(default_factory=OrderTimeInForce)
    entry_pricing: EntryPricing = Field(default_factory=EntryPricing)
    exit_pricing: ExitPricing = Field(default_factory=ExitPricing)
    
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    pairlists: list[PairlistConfig] = Field(default_factory=list)
    protections: list[ProtectionConfig] = Field(default_factory=list)
    
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    api_server: APIServerConfig = Field(default_factory=APIServerConfig)
    webhook: WebhookConfig = Field(default_factory=WebhookConfig)
    
    bot_name: str = Field(default="reco-trading")
    db_url: str = Field(default="sqlite:///reco_trading.db")
    initial_state: Literal["running", "stopped"] = Field(default="running")
    force_entry_enable: bool = Field(default=False)
    
    internals: dict = Field(default_factory=lambda: {"process_throttle_secs": 5, "heartbeat_interval": 60})
    
    strategy: str = Field(default="DefaultStrategy", description="Strategy name")
    strategy_path: str = Field(default="", description="Strategy path")
    
    dataformat_ohlcv: Literal["json", "jsongz", "feather", "parquet"] = Field(default="feather")
    dataformat_trades: Literal["json", "jsongz", "feather", "parquet"] = Field(default="feather")


class BacktestConfig(BaseModel):
    position_stacking: bool = Field(default=False)
    enable_protections: bool = Field(default=False)
    enable_dynamic_pairlist: bool = Field(default=False)
    timerange: str = Field(default="")
    timeframe_detail: str = Field(default="")
    export: Literal["none", "trades", "signals"] = Field(default="trades")
    exportfilename: str = Field(default="")
    backtest_breakdown: list[Literal["day", "week", "month", "year", "weekday"]] = Field(default_factory=list)
    backtest_cache: Literal["none", "day", "week", "month"] = Field(default="day")
    backtest_notes: str = Field(default="")


class HyperoptConfig(BaseModel):
    epochs: int = Field(default=100, ge=1, description="Number of epochs")
    spaces: list[str] = Field(default_factory=lambda: ["default"])
    print_all: bool = Field(default=False)
    print_json: bool = Field(default=False)
    hyperopt_jobs: int = Field(default=1)
    hyperopt_random_state: int | None = Field(default=None)
    hyperopt_min_trades: int = Field(default=10)
    hyperopt_loss: str = Field(default="OnlyProfitHyperOptLoss")
    disableparamexport: bool = Field(default=False)
    early_stop: int | None = Field(default=None)


def validate_config(config: dict[str, Any]) -> BotConfig:
    """
    Validate and return a BotConfig instance from a config dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validated BotConfig instance
        
    Raises:
        ValidationError: If configuration is invalid
    """
    return BotConfig(**config)


def get_default_config() -> dict[str, Any]:
    """
    Returns the default configuration for Reco-Trading.
    
    Returns:
        Default configuration dictionary
    """
    return {
        "max_open_trades": 3,
        "stake_currency": "USDT",
        "stake_amount": 0.05,
        "tradable_balance_ratio": 0.99,
        "fiat_display_currency": "USD",
        "dry_run": True,
        "dry_run_wallet": DRY_RUN_WALLET,
        "timeframe": "5m",
        "trailing_stop": False,
        "use_exit_signal": True,
        "exit_profit_only": False,
        "trading_mode": "spot",
        "minimal_roi": {
            "0": 0.10,
            "30": 0.05,
            "60": 0.02,
        },
        "stoploss": -0.10,
        "unfilledtimeout": {
            "entry": 10,
            "exit": 10,
            "unit": "minutes",
        },
        "order_types": {
            "entry": "limit",
            "exit": "limit",
            "stoploss": "market",
            "stoploss_on_exchange": False,
        },
        "exchange": {
            "name": "binance",
            "key": "",
            "secret": "",
            "pair_whitelist": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        },
        "pairlists": [
            {"method": "StaticPairList"},
            {"method": "VolumePairList", "config": {"number_assets": 10}},
            {"method": "PriceFilter", "config": {"min_price": 0.00000010}},
        ],
        "api_server": {
            "enabled": False,
            "listen_ip_address": "127.0.0.1",
            "listen_port": 8080,
        },
        "telegram": {
            "enabled": False,
        },
        "strategy": "DefaultStrategy",
        "db_url": "sqlite:///reco_trading.db",
    }
