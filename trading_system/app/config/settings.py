import hashlib

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore', populate_by_name=True, env_prefix='TRADING_SYSTEM_')

    mode: str = Field(default='paper')
    app_env: str = Field(default='dev')
    symbol: str = Field(default='BTCUSDT')

    api_key: str = Field(default='', validation_alias=AliasChoices('api_key', 'binance_api_key'))
    api_secret: str = Field(default='', validation_alias=AliasChoices('api_secret', 'binance_api_secret'))
    testnet: bool = Field(default=True, validation_alias=AliasChoices('testnet', 'binance_testnet'))
    confirm_mainnet: bool = Field(default=False)

    enable_live_trading: bool = Field(default=False)
    live_ack_token: str = Field(default='')

    runtime_ip: str = Field(default='')
    allowed_ip_hash: str = Field(default='')

    binance_max_weight: int = Field(default=900)
    order_rate_per_sec: int = Field(default=8)

    max_drawdown: float = Field(default=0.15)
    max_consecutive_losses: int = Field(default=5)
    risk_per_trade: float = Field(default=0.01)

    postgres_dsn: str = Field(default='postgresql+asyncpg://trading:trading@localhost:5432/trading')
    redis_url: str = Field(default='redis://localhost:6379/0')

    @property
    def is_live_mode(self) -> bool:
        return self.mode == 'live'

    @property
    def runtime_ip_hash(self) -> str:
        if not self.runtime_ip:
            return ''
        return hashlib.sha256(self.runtime_ip.encode('utf-8')).hexdigest()

    @model_validator(mode='after')
    def validate_runtime_safety(self) -> 'Settings':
        self.app_env = self.app_env.strip().lower()
        self.mode = self.mode.strip().lower()

        if self.app_env not in {'dev', 'staging', 'prod'}:
            raise ValueError(f"Invalid app_env '{self.app_env}'. Allowed values: dev, staging, prod")
        if self.mode not in {'paper', 'live'}:
            raise ValueError('Invalid mode. Allowed values: paper, live')

        if self.mode == 'live' and (not self.api_key or not self.api_secret):
            raise ValueError('Invalid configuration: api_key and api_secret are required')

        if self.mode == 'live' and not self.enable_live_trading:
            raise ValueError('mode=live requires enable_live_trading=true')
        if self.mode == 'live' and self.live_ack_token != 'ENABLE_LIVE_TRADING':
            raise ValueError('mode=live requires live_ack_token=ENABLE_LIVE_TRADING')
        if self.mode == 'live' and self.app_env != 'prod':
            raise ValueError('mode=live requires app_env=prod')
        if self.mode == 'live' and self.testnet:
            raise ValueError('mode=live requires testnet=False')
        if self.mode == 'paper' and not self.testnet:
            raise ValueError('mode=paper requires testnet=True')

        if not self.testnet:
            if not self.allowed_ip_hash:
                raise ValueError('Security gate blocked: allowed_ip_hash is required for mainnet mode')
            if not self.runtime_ip:
                raise ValueError('Security gate blocked: runtime_ip is required for mainnet mode')
            if self.runtime_ip_hash != self.allowed_ip_hash:
                raise ValueError('Security gate blocked: runtime_ip does not match allowed_ip_hash')

        return self
