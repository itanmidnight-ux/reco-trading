from __future__ import annotations

from typing import Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings

from reco_trading.config.symbols import normalize_symbol


class Settings(BaseSettings):
    """Runtime configuration loaded from .env and environment variables."""

    # =========================
    # BINANCE CONFIGURATION
    # =========================
    binance_api_key: str = Field(default="")
    binance_api_secret: str = Field(default="")
    binance_testnet: bool = True
    confirm_mainnet: bool = False

    # =========================
    # ENVIRONMENT
    # =========================
    environment: str = "testnet"
    runtime_profile: str = "paper"

    # =========================
    # TRADING SETTINGS
    # =========================
    trading_symbol: str = Field(default="BTCUSDT", validation_alias=AliasChoices("TRADING_SYMBOL", "SYMBOL", "BASE_SYMBOL"))
    timeframe: str = Field(default="5m", validation_alias=AliasChoices("TIMEFRAME", "PRIMARY_TIMEFRAME"))
    confirmation_timeframe: str = Field(default="15m", validation_alias=AliasChoices("CONFIRMATION_TIMEFRAME"))
    loop_sleep_seconds: int = 15
    history_limit: int = 300

    # =========================
    # SIGNAL THRESHOLDS
    # =========================
    min_signal_confidence: float = Field(default=0.75, validation_alias=AliasChoices("MIN_SIGNAL_CONFIDENCE", "CONFIDENCE_THRESHOLD"))
    strong_signal_confidence: float = 0.85
    exceptional_signal_confidence: float = 0.90
    confidence_hold_threshold: float = 0.55
    adx_min_threshold: float = 20.0
    max_spread_ratio: float = 0.002
    min_volume_ratio: float = 0.7

    # =========================
    # RISK MANAGEMENT
    # =========================
    risk_per_trade_fraction: float = 0.01
    min_trade_usdt: float = 5.0
    max_concurrent_trades: int = 1
    max_trades_per_day: int = 10
    max_trade_balance_fraction: float = 0.20
    daily_loss_limit_fraction: float = 0.03
    max_drawdown_fraction: float = 0.10

    # =========================
    # COOLDOWNS
    # =========================
    cooldown_minutes: int = 10
    loss_pause_minutes: int = 60
    loss_pause_after_consecutive: int = 3

    # =========================
    # DATABASE
    # =========================
    postgres_dsn: str
    postgres_admin_dsn: Optional[str] = None

    # =========================
    # REDIS
    # =========================
    redis_url: str = "redis://localhost:6379/0"

    @property
    def symbol(self) -> str:
        return self.trading_symbol

    @property
    def primary_timeframe(self) -> str:
        return self.timeframe

    @property
    def confidence_threshold(self) -> float:
        return self.min_signal_confidence

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
