from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(slots=True)
class Settings:
    """Runtime configuration loaded from environment variables."""

    binance_api_key: str
    binance_api_secret: str
    postgres_dsn: str
    binance_testnet: bool
    environment: str
    runtime_profile: str
    symbol: str = "BTC/USDT"
    primary_timeframe: str = "5m"
    confirmation_timeframe: str = "15m"
    loop_sleep_seconds: int = 15
    history_limit: int = 300
    max_trades_per_day: int = 10
    max_trade_balance_fraction: float = 0.20
    confidence_threshold: float = 0.75
    daily_loss_limit_fraction: float = 0.03
    cooldown_minutes: int = 10

    @classmethod
    def from_env(cls) -> "Settings":
        def _bool(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            if raw is None:
                return default
            return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

        return cls(
            binance_api_key=os.getenv("BINANCE_API_KEY", ""),
            binance_api_secret=os.getenv("BINANCE_API_SECRET", ""),
            postgres_dsn=os.getenv("POSTGRES_DSN", ""),
            binance_testnet=_bool("BINANCE_TESTNET", True),
            environment=os.getenv("ENVIRONMENT", "testnet"),
            runtime_profile=os.getenv("RUNTIME_PROFILE", "paper"),
            cooldown_minutes=int(os.getenv("COOLDOWN_MINUTES", "10")),
            risk_per_trade_fraction=float(os.getenv("RISK_PER_TRADE_FRACTION", "0.02")),
            max_trades_per_hour=int(os.getenv("MAX_TRADES_PER_HOUR", "3")),
            adx_min_threshold=float(os.getenv("ADX_MIN_THRESHOLD", "20")),
            max_spread_ratio=float(os.getenv("MAX_SPREAD_RATIO", "0.0015")),
            min_volume_ratio=float(os.getenv("MIN_VOLUME_RATIO", "0.7")),
            loss_pause_after_consecutive=int(os.getenv("LOSS_PAUSE_AFTER_CONSECUTIVE", "3")),
            loss_pause_minutes=int(os.getenv("LOSS_PAUSE_MINUTES", "60")),
        )
