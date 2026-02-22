from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ALLOWED_STRATEGIES: tuple[str, str, str] = (
    'directional',
    'adaptive_market_making',
    'multi_exchange_arbitrage',
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', case_sensitive=False, extra='ignore')

    environment: str = Field(default='production')
    runtime_profile: str = Field(default='production')
    symbol: str = Field(default='BTC/USDT', validation_alias='BASE_SYMBOL')
    timeframe: str = Field(default='1m', validation_alias='TIMEFRAME')

    binance_api_key: SecretStr = Field(validation_alias='BINANCE_API_KEY')
    binance_api_secret: SecretStr = Field(validation_alias='BINANCE_API_SECRET')
    encrypted_api_key: str | None = None
    encrypted_api_secret: str | None = None
    encryption_key: SecretStr | None = None
    binance_testnet: bool = Field(default=True, validation_alias='BINANCE_TESTNET')
    confirm_mainnet: bool = False

    postgres_dsn: str = Field(
        ...,
        description='Cadena de conexión PostgreSQL obligatoria. Defínela vía POSTGRES_DSN en .env.',
    )
    postgres_admin_dsn: str | None = Field(default=None)
    redis_url: str = Field(default='redis://localhost:6379/0')
    broker_backend: str = Field(default='redis')
    broker_stream_prefix: str = Field(default='reco:broker')
    broker_topic_prefix: str = Field(default='reco.broker')
    kafka_bootstrap_servers: str = Field(default='localhost:9092')
    broker_max_retries: int = Field(default=3, ge=0, le=20)
    broker_retry_backoff_seconds: float = Field(default=0.5, ge=0.0, le=60.0)
    broker_retry_backoff_max_seconds: float = Field(default=30.0, ge=0.1, le=600.0)
    broker_consume_block_ms: int = Field(default=5000, ge=1, le=60000)
    broker_operation_timeout_seconds: float = Field(default=5.0, ge=0.1, le=120.0)
    execution_order_timeout_seconds: float = Field(default=30.0, ge=0.1, le=300.0)
    broker_stream_maxlen: int = Field(default=50_000, ge=100, le=5_000_000)
    broker_dlq_maxlen: int = Field(default=20_000, ge=100, le=5_000_000)
    broker_retention_ms: int = Field(default=86_400_000, ge=60_000, le=2_592_000_000)
    broker_idempotency_ttl_seconds: int = Field(default=86_400, ge=60, le=2_592_000)

    risk_per_trade: float = Field(default=0.01, ge=0.001, le=0.05, validation_alias='RISK_PER_TRADE')
    max_daily_drawdown: float = Field(default=0.03, ge=0.01, le=0.2)
    max_daily_loss: float = Field(default=0.02, ge=0.005, le=0.2)
    max_global_drawdown: float = Field(default=0.12, ge=0.02, le=0.6)
    max_total_exposure: float = Field(default=0.7, ge=0.1, le=1.0)
    max_asset_exposure: float = Field(default=0.35, ge=0.05, le=1.0)
    correlation_threshold: float = Field(default=0.75, ge=0.1, le=0.99)

    max_consecutive_losses: int = Field(default=3, ge=1, le=10)
    atr_stop_multiplier: float = Field(default=2.0, ge=1.0, le=6.0)
    volatility_target: float = Field(default=0.20, ge=0.01, le=2.0)
    circuit_breaker_volatility: float = Field(default=0.08, ge=0.01, le=1.0)

    maker_fee: float = Field(default=0.001)
    taker_fee: float = Field(default=0.001)
    slippage_bps: float = Field(default=5.0)

    conservative_mode_enabled: bool = Field(default=True)
    enable_directional_strategy: bool = Field(default=True)
    enable_adaptive_market_making: bool = Field(default=True)
    enable_multi_exchange_arbitrage: bool = Field(default=False)

    learning_phase_seconds: int = Field(default=300, ge=60, le=3600)
    confidence_hold_threshold: float = Field(default=0.60, ge=0.50, le=0.95)
    confidence_tier_1: float = Field(default=0.65, ge=0.50, le=0.99)
    confidence_tier_2: float = Field(default=0.70, ge=0.50, le=0.99)
    confidence_tier_3: float = Field(default=0.80, ge=0.50, le=0.99)
    confidence_tier_4: float = Field(default=0.90, ge=0.50, le=0.99)
    confidence_alloc_tier_1: float = Field(default=0.0025, ge=0.0, le=0.05)
    confidence_alloc_tier_2: float = Field(default=0.0050, ge=0.0, le=0.05)
    confidence_alloc_tier_3: float = Field(default=0.0100, ge=0.0, le=0.05)
    confidence_alloc_tier_4: float = Field(default=0.0200, ge=0.0, le=0.05)
    max_confidence_allocation: float = Field(default=0.02, ge=0.001, le=0.05)
    target_scalp_seconds: int = Field(default=20, ge=5, le=120)
    max_position_seconds: int = Field(default=30, ge=10, le=300)

    loop_interval_seconds: int = Field(default=5, ge=1, le=60)

    monitoring_metrics_enabled: bool = Field(default=True)
    monitoring_metrics_host: str = Field(default='0.0.0.0')
    monitoring_metrics_port: int = Field(default=8001, ge=1, le=65535)

    runtime_enable_uvloop: bool = Field(default=True)
    runtime_min_nofile: int = Field(default=4096, ge=256, le=1_048_576)
    runtime_target_nofile: int = Field(default=65_535, ge=1024, le=1_048_576)
    runtime_cpu_affinity: str | None = Field(default=None, description='Lista de CPUs separada por comas, ejemplo: 0,1,2')



    @field_validator('binance_api_key', 'binance_api_secret')
    @classmethod
    def non_empty_binance_credentials(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError('Las credenciales de Binance no pueden estar vacías.')
        return value

    @field_validator('broker_backend')
    @classmethod
    def validate_broker_backend(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in {'redis', 'kafka'}:
            raise ValueError('broker_backend debe ser "redis" o "kafka".')
        return normalized

    @field_validator('symbol')
    @classmethod
    def validate_symbol_format(cls, value: str) -> str:
        normalized = value.strip().upper()
        if '/' not in normalized:
            raise ValueError('symbol/base_symbol debe tener formato BASE/QUOTE (ej: BTC/USDT).')
        base, quote = normalized.split('/', 1)
        if not base or not quote:
            raise ValueError('symbol/base_symbol inválido.')
        return normalized

    @field_validator('timeframe')
    @classmethod
    def validate_timeframe(cls, value: str) -> str:
        normalized = value.strip().lower()
        allowed = {'1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d'}
        if normalized not in allowed:
            raise ValueError(f'timeframe inválido: {normalized}.')
        return normalized

    @model_validator(mode='after')
    def validate_mainnet_guardrail(self) -> 'Settings':
        if not self.binance_testnet and not self.confirm_mainnet:
            raise ValueError('Mainnet requiere confirm_mainnet=true explícito por seguridad institucional.')
        return self

    @model_validator(mode='after')
    def validate_strategy_toggles(self) -> 'Settings':
        if not self.enabled_strategies:
            raise ValueError('Debe existir al menos una estrategia habilitada en el kernel.')
        return self

    @computed_field
    @property
    def enabled_strategies(self) -> tuple[str, ...]:
        enabled: list[str] = []
        if self.enable_directional_strategy:
            enabled.append('directional')
        if self.enable_adaptive_market_making:
            enabled.append('adaptive_market_making')
        if self.enable_multi_exchange_arbitrage:
            enabled.append('multi_exchange_arbitrage')
        return tuple(enabled)

    @computed_field
    @property
    def symbol_rest(self) -> str:
        return self.symbol.replace('/', '')


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
