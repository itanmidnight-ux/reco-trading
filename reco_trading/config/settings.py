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
    low_ram_mode: bool = True
    max_ram_mb: int = 500
    ui_state_emit_on_each_log: bool = False
    # LLM settings disabled - using filter-based validation only
    # llm_fix_cycle_interval_seconds: int = 3600
    # llm_cleanup_interval_seconds: int = 86400
    # llm_mode: str = "base"
    # llm_local_model: str = "qwen2.5:0.5b"
    # ollama_base_url: str = "http://localhost:11434"
    # llm_remote_endpoint: str = "https://api.openai.com/v1/chat/completions"
    # llm_remote_model: str = "gpt-4o-mini"
    # llm_remote_api_key: str = ""
    # llm_local_keep_alive: str = "20m"
    # llm_local_num_ctx: int = 256
    # llm_local_num_predict: int = 4
    # llm_local_top_p: float = 0.9
    # llm_local_temperature: float = 0.0
    # llm_local_healthcheck_enabled: bool = True

    # =========================
    # SIGNAL THRESHOLDS
    # =========================
    min_signal_confidence: float = Field(default=0.70, validation_alias=AliasChoices("MIN_SIGNAL_CONFIDENCE", "CONFIDENCE_THRESHOLD"))
    strong_signal_confidence: float = 0.80
    exceptional_signal_confidence: float = 0.85
    confidence_hold_threshold: float = 0.70
    adx_min_threshold: float = 15.0
    max_spread_ratio: float = 0.004
    max_slippage_ratio: float = 0.003
    min_volume_ratio: float = 0.7
    execution_model_enabled: bool = False
    auto_confidence_adjustment: bool = True

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
    min_expected_reward_risk: float = 1.8
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 4.0
    auto_stop_enabled: bool = True
    auto_stop_break_even_trigger_pct: float = 0.008
    auto_stop_break_even_buffer_pct: float = 0.0005
    auto_stop_trailing_activate_pct: float = 0.012
    auto_stop_trailing_delta_low_vol_pct: float = 0.006
    auto_stop_trailing_delta_high_vol_pct: float = 0.010
    auto_stop_max_duration_minutes: int = 180

    # =========================
    # COOLDOWNS
    # =========================
    cooldown_minutes: int = 4
    loss_pause_minutes: int = 20
    loss_pause_after_consecutive: int = 4

    # =========================
    # DATABASE (PostgreSQL, MySQL, or SQLite)
    # =========================
    postgres_dsn: Optional[str] = None
    postgres_admin_dsn: Optional[str] = None
    mysql_dsn: Optional[str] = None
    database_url: Optional[str] = None

    # =========================
    # REDIS
    # =========================
    redis_url: str = "redis://localhost:6379/0"
    observability_enabled: bool = True
    verbose_trade_decision_logs: bool = False
    observability_bind_host: str = "0.0.0.0"
    observability_port: int = 9108
    api_latency_window_size: int = 200
    stale_market_data_max_age_seconds: int = 180
    feature_multi_symbol_enabled: bool = False
    trading_symbols: list[str] = Field(default_factory=lambda: [
        "BTC/USDT", "ETH/USDT"
    ], validation_alias=AliasChoices("TRADING_SYMBOLS", "SYMBOLS"))
    max_global_exposure_fraction: float = 0.7
    max_symbol_correlation: float = 0.85
    symbol_capital_limits: dict[str, float] = Field(default_factory=dict, validation_alias=AliasChoices("SYMBOL_CAPITAL_LIMITS"))

    # =========================
    # FUTURES TRADING
    # =========================
    futures_enabled: bool = False
    futures_max_leverage: int = 50
    default_leverage: int = 10
    enable_position_hedging: bool = False

    # =========================
    # MOBILE APP
    # =========================
    mobile_api_enabled: bool = False
    mobile_ws_enabled: bool = False
    mobile_push_notifications: bool = False

    # =========================
    # SOCIAL TRADING
    # =========================
    social_trading_enabled: bool = False
    strategy_marketplace: bool = False
    copy_trading_enabled: bool = False

    # =========================
    # GRID TRADING
    # =========================
    grid_trading_enabled: bool = False
    atr_grid_spacing: bool = True
    dynamic_rebalancing: bool = True

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

    @field_validator("binance_api_key", "binance_api_secret", mode="before")
    @classmethod
    def _sanitize_api_credentials(cls, value: object) -> str:
        if value is None:
            return ""
        secret = str(value).strip()
        lowered = secret.lower()
        if lowered in {"", "none", "null", "changeme", "your_api_key_here", "your_api_secret_here"}:
            return ""
        return secret

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
        case_sensitive=False,
    )
