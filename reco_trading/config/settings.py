from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime configuration loaded from .env and environment variables."""

    # Binance
    binance_api_key: str = Field(default="")
    binance_api_secret: str = Field(default="")
    binance_testnet: bool = True
    confirm_mainnet: bool = False

    # Environment
    environment: str = "testnet"
    runtime_profile: str = "paper"

    # Trading configuration
    symbol: str = Field(default="BTC/USDT", validation_alias=AliasChoices("SYMBOL", "BASE_SYMBOL", "TRADING_SYMBOL"))
    primary_timeframe: str = Field(default="5m", validation_alias=AliasChoices("PRIMARY_TIMEFRAME", "TIMEFRAME"))
    confirmation_timeframe: str = Field(default="15m", validation_alias=AliasChoices("CONFIRMATION_TIMEFRAME"))
    loop_sleep_seconds: int = 15
    history_limit: int = 300

    # Risk management
    max_trades_per_day: int = 10
    max_trade_balance_fraction: float = 0.20
    risk_per_trade_fraction: float = 0.01
    min_trade_usdt: float = 5.0
    max_concurrent_trades: int = 1
    daily_loss_limit_fraction: float = 0.03

    # Strategy thresholds
    confidence_threshold: float = Field(default=0.75, validation_alias=AliasChoices("CONFIDENCE_THRESHOLD", "MIN_SIGNAL_CONFIDENCE"))
    strong_signal_confidence: float = 0.85
    exceptional_signal_confidence: float = 0.90
    adx_min_threshold: float = 20.0
    max_spread_ratio: float = 0.0015
    min_volume_ratio: float = 0.7

    # Cooldowns
    cooldown_minutes: int = 10
    loss_pause_after_consecutive: int = 3
    loss_pause_minutes: int = 60

    # Infrastructure
    postgres_dsn: str = ""
    redis_url: str = "redis://localhost:6379/0"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
