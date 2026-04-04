from __future__ import annotations

import asyncio
import logging
import sys
import threading

from reco_trading.config.settings import Settings
from reco_trading.core.bot_engine import BotEngine
from reco_trading.database.repository import Repository

_bot_instance: BotEngine | None = None
_bot_runtime_error: Exception | None = None


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def get_bot_instance() -> BotEngine | None:
    return _bot_instance


def _run_bot(settings: Settings, state_manager: object | None) -> None:
    global _bot_instance, _bot_runtime_error
    bot = BotEngine(settings, state_manager=state_manager)
    _bot_instance = bot
    try:
        asyncio.run(bot.run())
    except Exception as exc:  # noqa: BLE001
        _bot_runtime_error = exc
        logging.getLogger(__name__).exception("Bot terminated unexpectedly")
    finally:
        _bot_instance = None


def _join_bot_thread_or_exit(bot_thread: threading.Thread, logger: logging.Logger) -> None:
    bot_thread.join()
    if _bot_runtime_error is not None:
        logger.error("Bot stopped unexpectedly: %s", _bot_runtime_error)
        sys.exit(1)


async def _verify_database_connection(settings: Settings) -> str:
    repository = Repository(settings.postgres_dsn)
    try:
        await repository.verify_connectivity()
        return settings.postgres_dsn
    finally:
        await repository.close()


def run() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)

    global _bot_runtime_error
    _bot_runtime_error = None

    settings = Settings()
    if not settings.postgres_dsn:
        raise RuntimeError("POSTGRES_DSN is required")
    if settings.require_api_keys and (not settings.binance_api_key or not settings.binance_api_secret):
        raise RuntimeError("BINANCE_API_KEY and BINANCE_API_SECRET are required")
    if not settings.binance_testnet and not settings.confirm_mainnet:
        raise RuntimeError("Mainnet trading blocked: set CONFIRM_MAINNET=true to proceed")

    try:
        asyncio.run(_verify_database_connection(settings))
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Database unavailable; start PostgreSQL or fix POSTGRES_DSN before launching the bot: %s",
            exc,
        )
        raise SystemExit(1) from None

    try:
        from reco_trading.ui import StateManager, run_gui

        state_manager = StateManager()
    except Exception as exc:  # noqa: BLE001
        logger.exception("UI initialization failed, running bot headless: %s", exc)
        _run_bot(settings, None)
        return

    try:
        from reco_trading.ui.bootstrap import hydrate_state_from_database

        asyncio.run(hydrate_state_from_database(settings, state_manager))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "State hydration failed; continuing with an empty UI state because the bot can still start cleanly: %s",
            exc,
        )

    bot_thread = threading.Thread(target=_run_bot, args=(settings, state_manager), daemon=True, name="bot-engine")
    bot_thread.start()

    try:
        run_gui(state_manager)
    except Exception as exc:  # noqa: BLE001
        logger.exception("GUI failed, bot will continue running: %s", exc)
        _join_bot_thread_or_exit(bot_thread, logger)


if __name__ == "__main__":
    run()
