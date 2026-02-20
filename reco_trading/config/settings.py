from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', case_sensitive=False)

    environment: str = Field(default='production')
    symbol: str = Field(default='BTC/USDT')
    timeframe: str = Field(default='5m')

    binance_api_key: SecretStr
    binance_api_secret: SecretStr
    binance_testnet: bool = False

    postgres_dsn: str = Field(default='postgresql+asyncpg://trading:trading@localhost:5432/trading')
    postgres_admin_dsn: str | None = Field(default=None)
    redis_url: str = Field(default='redis://localhost:6379/0')

    risk_per_trade: float = Field(default=0.01, ge=0.001, le=0.05)
    max_daily_drawdown: float = Field(default=0.03, ge=0.01, le=0.2)
    max_consecutive_losses: int = Field(default=3, ge=1, le=10)
    atr_stop_multiplier: float = Field(default=2.0, ge=1.0, le=6.0)
    volatility_target: float = Field(default=0.20, ge=0.01, le=2.0)
    circuit_breaker_volatility: float = Field(default=0.08, ge=0.01, le=1.0)

    maker_fee: float = Field(default=0.001)
    taker_fee: float = Field(default=0.001)
    slippage_bps: float = Field(default=5.0)

    loop_interval_seconds: int = Field(default=5, ge=1, le=60)

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

    @computed_field
    @property
    def symbol_rest(self) -> str:
        return self.symbol.replace('/', '')


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
