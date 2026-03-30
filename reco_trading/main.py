from __future__ import annotations

import asyncio
import logging
import sys
import threading

from reco_trading.config.settings import Settings
from reco_trading.core.bot_engine import BotEngine
from reco_trading.database.repository import Repository


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def _run_bot(settings: Settings, state_manager: object | None) -> None:
    """Run the trading bot."""
    try:
        asyncio.run(BotEngine(settings, state_manager=state_manager).run())
    except Exception as e:
        logging.getLogger(__name__).error(f"Bot error: {e}")
        raise


async def _verify_database_connection(settings: Settings) -> None:
    """Verify database connection."""
    repository = Repository(settings.postgres_dsn)
    try:
        await repository.verify_connectivity()
    finally:
        await repository.close()


def run() -> None:
    """Main entry point."""
    configure_logging()
    logger = logging.getLogger(__name__)
    
    settings = Settings()
    
    # Verify required settings
    if not settings.postgres_dsn:
        logger.error("POSTGRES_DSN is required")
        sys.exit(1)
    
    if settings.require_api_keys and (not settings.binance_api_key or not settings.binance_api_secret):
        logger.error("BINANCE_API_KEY and BINANCE_API_SECRET are required")
        sys.exit(1)
    
    if not settings.binance_testnet and not settings.confirm_mainnet:
        logger.error("Mainnet trading blocked: set CONFIRM_MAINNET=true to proceed")
        sys.exit(1)
    
    # Verify database connection
    try:
        asyncio.run(_verify_database_connection(settings))
    except Exception as exc:
        logger.error(
            "Database unavailable; start PostgreSQL or fix POSTGRES_DSN before launching the bot: %s",
            exc,
        )
        sys.exit(1)

    # Try to start UI, fallback to headless
    state_manager = None
    try:
        from reco_trading.ui import StateManager, run_gui

        state_manager = StateManager()
    except Exception as exc:
        logger.exception("UI initialization failed, running bot headless: %s", exc)
        _run_bot(settings, None)
        return

    # Hydrate state from database
    try:
        from reco_trading.ui.bootstrap import hydrate_state_from_database

        asyncio.run(hydrate_state_from_database(settings, state_manager))
    except Exception as exc:
        logger.warning(
            "State hydration failed; continuing with an empty UI state because the bot can still start cleanly: %s",
            exc,
        )

    # Start bot in background thread
    bot_thread = threading.Thread(
        target=_run_bot, 
        args=(settings, state_manager), 
        daemon=True, 
        name="bot-engine"
    )
    bot_thread.start()

    # Run GUI
    try:
        run_gui(state_manager)
    except Exception as exc:
        logger.exception("GUI failed, bot will continue running: %s", exc)
        bot_thread.join()


if __name__ == "__main__":
    run()
