import hashlib

from pydantic import Field
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

    app_env: str = Field(default='dev')
    mode: str = Field(default='paper')
    symbol: str = Field(default='BTCUSDT')

    api_key: str = Field(default='')
    api_secret: str = Field(default='')
    testnet: bool = Field(default=True)

    enable_live_trading: bool = Field(default=False)
    live_ack_token: str = Field(default='')
    live_ack_token_expected: str = Field(default='ENABLE_LIVE_TRADING')
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
        self.mode = self.mode.strip().lower()
        self.app_env = self.app_env.strip().lower()

        if self.mode not in {'paper', 'live'}:
            raise ValueError(f"Invalid mode '{self.mode}'. Allowed values: paper, live")

        if self.app_env not in {'dev', 'staging', 'prod'}:
            raise ValueError(f"Invalid app_env '{self.app_env}'. Allowed values: dev, staging, prod")

        if self.is_live_mode:
            if self.testnet:
                raise ValueError('Invalid configuration: mode=live requires testnet=False')
            if self.app_env != 'prod':
                raise ValueError('Invalid configuration: mode=live requires app_env=prod')
            if not self.api_key or not self.api_secret:
                raise ValueError('Invalid configuration: mode=live requires api_key and api_secret')
            if not self.enable_live_trading:
                raise ValueError('Security gate blocked: set enable_live_trading=true to allow live mode')
            if self.live_ack_token != self.live_ack_token_expected:
                raise ValueError('Security gate blocked: live_ack_token is invalid or missing')
            if not self.allowed_ip_hash:
                raise ValueError('Security gate blocked: allowed_ip_hash is required for live mode')
            if not self.runtime_ip:
                raise ValueError('Security gate blocked: runtime_ip is required for live mode')
            if self.runtime_ip_hash != self.allowed_ip_hash:
                raise ValueError('Security gate blocked: runtime_ip does not match allowed_ip_hash')

        if self.mode == 'paper' and not self.testnet:
            raise ValueError('Invalid configuration: mode=paper requires testnet=True')

        return self
