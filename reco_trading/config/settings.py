from __future__ import annotations

from typing import Optional

from pydantic import AliasChoices, Field, ConfigDict, field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Runtime configuration loaded from .env and environment variables."""

    # =========================
    # BINANCE CONFIGURATION
    # =========================
    binance_api_key: str = Field(default="")
    binance_api_secret: str = Field(default="")
    binance_testnet: bool = True
    confirm_mainnet: bool = False
    require_api_keys: bool = True

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
    min_signal_confidence: float = Field(default=0.62, validation_alias=AliasChoices("MIN_SIGNAL_CONFIDENCE", "CONFIDENCE_THRESHOLD"))
    strong_signal_confidence: float = 0.85
    exceptional_signal_confidence: float = 0.90
    confidence_hold_threshold: float = 0.55
    adx_min_threshold: float = 20.0
    max_spread_ratio: float = 0.0025
    max_slippage_ratio: float = 0.003
    min_volume_ratio: float = 0.9
    execution_model_enabled: bool = False

    # =========================
    # MARKET INTELLIGENCE
    # =========================
    enable_market_intelligence: bool = True
    volatility_filter_enabled: bool = True
    liquidity_zone_filter_enabled: bool = True
    market_regime_classifier_enabled: bool = True
    market_range_filter_enabled: bool = True
    liquidity_proximity_threshold: float = 0.010

    # =========================
    # RISK MANAGEMENT
    # =========================
    risk_per_trade_fraction: float = 0.01
    min_trade_usdt: float = 5.0
    max_concurrent_trades: int = 1
    max_trades_per_day: int = 10
    max_trade_balance_fraction: float = 0.20
    spot_only_mode: bool = True
    daily_loss_limit_fraction: float = 0.03
    max_drawdown_fraction: float = 0.10
    capital_reserve_ratio: float = 0.15
    min_cash_buffer_usdt: float = 10.0
    enable_capital_profiles: bool = True
    enforce_fee_floor: bool = True
    estimated_fee_rate: float = 0.001
    min_expected_reward_risk: float = 2.0

    # =========================
    # COOLDOWNS
    # =========================
    cooldown_minutes: int = 4
    loss_pause_minutes: int = 20
    loss_pause_after_consecutive: int = 4

    # =========================
    # DATABASE
    # =========================
    postgres_dsn: str
    postgres_admin_dsn: Optional[str] = None

    # =========================
    # REDIS
    # =========================
    redis_url: str = "redis://localhost:6379/0"
    observability_enabled: bool = True
    observability_bind_host: str = "0.0.0.0"
    observability_port: int = 9108
    api_latency_window_size: int = 200
    stale_market_data_max_age_seconds: int = 180
    feature_multi_symbol_enabled: bool = False
    trading_symbols: list[str] = Field(default_factory=list, validation_alias=AliasChoices("TRADING_SYMBOLS", "SYMBOLS"))
    max_global_exposure_fraction: float = 0.7
    max_symbol_correlation: float = 0.85
    symbol_capital_limits: dict[str, float] = Field(default_factory=dict, validation_alias=AliasChoices("SYMBOL_CAPITAL_LIMITS"))

    @property
    def symbol(self) -> str:
        return self.trading_symbol

    @property
    def primary_timeframe(self) -> str:
        return self.timeframe

    @property
    def confidence_threshold(self) -> float:
        return self.min_signal_confidence

    @field_validator("trading_symbol")
    @classmethod
    def _normalize_symbol(cls, value: str) -> str:
        return str(value or "BTCUSDT").replace("/", "").upper()

    @field_validator("trading_symbols", mode="before")
    @classmethod
    def _parse_symbols(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            raw = [item.strip() for item in value.split(",")]
        elif isinstance(value, list):
            raw = [str(item).strip() for item in value]
        else:
            return []
        return [item.replace("/", "").upper() for item in raw if item]

    @field_validator("observability_port")
    @classmethod
    def _validate_observability_port(cls, value: int) -> int:
        if value < 1 or value > 65535:
            raise ValueError("observability_port must be in range [1, 65535]")
        return value

    @field_validator(
        "risk_per_trade_fraction",
        "max_global_exposure_fraction",
        "max_symbol_correlation",
        "capital_reserve_ratio",
        "daily_loss_limit_fraction",
        "max_drawdown_fraction",
        "max_trade_balance_fraction",
    )
    @classmethod
    def _validate_fraction(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("fraction-like values must be within [0, 1]")
        return float(value)

    @field_validator("symbol_capital_limits", mode="before")
    @classmethod
    def _validate_symbol_capital_limits(cls, value: object) -> dict[str, float]:
        if not isinstance(value, dict):
            return {}
        normalized: dict[str, float] = {}
        for symbol, cap in value.items():
            try:
                cap_f = float(cap)
            except (TypeError, ValueError):
                continue
            if cap_f <= 0:
                continue
            normalized[str(symbol).replace("/", "").upper()] = cap_f
        return normalized
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )
