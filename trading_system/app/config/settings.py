import hashlib

from pydantic import Field
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

    app_env: str = Field(default='dev')
    symbol: str = Field(default='BTCUSDT')

    binance_api_key: str = Field(default='')
    binance_api_secret: str = Field(default='')
    binance_testnet: bool = Field(default=True)
    confirm_mainnet: bool = Field(default=False)

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
        return True

    @property
    def runtime_ip_hash(self) -> str:
        if not self.runtime_ip:
            return ''
        return hashlib.sha256(self.runtime_ip.encode('utf-8')).hexdigest()

    @model_validator(mode='after')
    def validate_runtime_safety(self) -> 'Settings':
        self.app_env = self.app_env.strip().lower()

        if self.app_env not in {'dev', 'staging', 'prod'}:
            raise ValueError(f"Invalid app_env '{self.app_env}'. Allowed values: dev, staging, prod")

        if not self.binance_api_key or not self.binance_api_secret:
            raise ValueError('Invalid configuration: binance_api_key and binance_api_secret are required')

        if not self.binance_testnet and not self.confirm_mainnet:
            raise ValueError('Mainnet requires confirm_mainnet=true')

        if not self.binance_testnet:
            if not self.allowed_ip_hash:
                raise ValueError('Security gate blocked: allowed_ip_hash is required for mainnet mode')
            if not self.runtime_ip:
                raise ValueError('Security gate blocked: runtime_ip is required for mainnet mode')
            if self.runtime_ip_hash != self.allowed_ip_hash:
                raise ValueError('Security gate blocked: runtime_ip does not match allowed_ip_hash')

        return self
