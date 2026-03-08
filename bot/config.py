from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BotConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', case_sensitive=False)

    api_key: str = Field(default='', alias='BINANCE_API_KEY')
    api_secret: str = Field(default='', alias='BINANCE_API_SECRET')
    testnet: bool = Field(default=True, alias='BINANCE_TESTNET')

    symbol: str = Field(default='BTC/USDT', alias='BOT_SYMBOL')
    timeframe: str = Field(default='15m', alias='BOT_TIMEFRAME')
    loop_interval_seconds: int = Field(default=60, alias='BOT_LOOP_SECONDS')

    max_risk_per_trade: float = 0.01
    max_trades_per_day: int = 3
    daily_loss_limit: float = 0.03
    max_open_positions: int = 1

    maker_fee: float = 0.001
    taker_fee: float = 0.001
    min_expected_edge_bps: float = 25.0

    order_quote_size_usd: float = Field(default=6.0, alias='BOT_ORDER_QUOTE_SIZE')
    fast_sma_period: int = 20
    slow_sma_period: int = 50
    momentum_lookback: int = 8
    rsi_period: int = 14
    min_volatility_pct: float = 0.003
    max_volatility_pct: float = 0.04

    recv_window: int = 10_000
    request_timeout_ms: int = 20_000
    retries: int = 5

    @property
    def exchange_name(self) -> str:
        return 'binance'


__all__ = ['BotConfig']
