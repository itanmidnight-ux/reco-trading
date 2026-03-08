from __future__ import annotations

import logging

from reco_trading.config.settings import Settings
from reco_trading.core.bot_engine import BotEngine


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s state=%(message)s",
    )


async def run() -> None:
    configure_logging()
    settings = Settings.from_env()
    if not settings.postgres_dsn:
        raise RuntimeError("POSTGRES_DSN is required")
    bot = BotEngine(settings)
    await bot.run()
