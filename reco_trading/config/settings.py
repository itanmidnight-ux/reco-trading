from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', case_sensitive=False)

    environment: str = Field(default='production')
    symbol: str = Field(default='BTC/USDT')
    timeframe: str = Field(default='5m')

    binance_api_key: SecretStr
    binance_api_secret: SecretStr
    encrypted_api_key: str | None = None
    encrypted_api_secret: str | None = None
    encryption_key: SecretStr | None = None
    binance_testnet: bool = True
    confirm_mainnet: bool = False

    postgres_dsn: str = Field(default='postgresql+asyncpg://trading:trading_password@localhost/reco_trading_prod')
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
    broker_stream_maxlen: int = Field(default=50_000, ge=100, le=5_000_000)
    broker_dlq_maxlen: int = Field(default=20_000, ge=100, le=5_000_000)
    broker_retention_ms: int = Field(default=86_400_000, ge=60_000, le=2_592_000_000)
    broker_idempotency_ttl_seconds: int = Field(default=86_400, ge=60, le=2_592_000)

    risk_per_trade: float = Field(default=0.01, ge=0.001, le=0.05)
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

    loop_interval_seconds: int = Field(default=5, ge=1, le=60)

    monitoring_metrics_enabled: bool = Field(default=True)
    monitoring_metrics_host: str = Field(default='0.0.0.0')
    monitoring_metrics_port: int = Field(default=8001, ge=1, le=65535)


    @field_validator('broker_backend')
    @classmethod
    def validate_broker_backend(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in {'redis', 'kafka'}:
            raise ValueError('broker_backend debe ser "redis" o "kafka".')
        return normalized

    @field_validator('symbol')
    @classmethod
    def only_btcusdt(cls, value: str) -> str:
        if value != 'BTC/USDT':
            raise ValueError('Solo se permite BTC/USDT para este deployment.')
        return value

    @field_validator('timeframe')
    @classmethod
    def only_5m(cls, value: str) -> str:
        if value != '5m':
            raise ValueError('Solo se permite timeframe 5m.')
        return value

    @model_validator(mode='after')
    def validate_mainnet_guardrail(self) -> 'Settings':
        if not self.binance_testnet and not self.confirm_mainnet:
            raise ValueError('Mainnet requiere confirm_mainnet=true explícito por seguridad institucional.')
        return self

    @computed_field
    @property
    def symbol_rest(self) -> str:
        return self.symbol.replace('/', '')


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
