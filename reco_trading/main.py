from __future__ import annotations

import asyncio
import logging
import threading

from reco_trading.config.settings import Settings
from reco_trading.core.bot_engine import BotEngine


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def _run_bot(settings: Settings, state_manager: object | None) -> None:
    asyncio.run(BotEngine(settings, state_manager=state_manager).run())


def run() -> None:
    configure_logging()
    settings = Settings()
    if not settings.postgres_dsn:
        raise RuntimeError("POSTGRES_DSN is required")
    if settings.require_api_keys and (not settings.binance_api_key or not settings.binance_api_secret):
        raise RuntimeError("BINANCE_API_KEY and BINANCE_API_SECRET are required")
    if not settings.binance_testnet and not settings.confirm_mainnet:
        raise RuntimeError("Mainnet trading blocked: set CONFIRM_MAINNET=true to proceed")
    if settings.environment.lower() == "production" and settings.binance_testnet:
        raise RuntimeError("ENVIRONMENT=production requires BINANCE_TESTNET=false")
    if settings.runtime_profile.lower() == "live" and settings.binance_testnet:
        raise RuntimeError("RUNTIME_PROFILE=live requires BINANCE_TESTNET=false")

    try:
        from reco_trading.ui import StateManager, run_gui
        from reco_trading.ui.bootstrap import hydrate_state_from_database

        state_manager = StateManager()
        asyncio.run(hydrate_state_from_database(settings, state_manager))
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).exception("UI initialization failed, running bot headless: %s", exc)
        _run_bot(settings, None)
        return

    bot_thread = threading.Thread(target=_run_bot, args=(settings, state_manager), daemon=True, name="bot-engine")
    bot_thread.start()

    try:
        run_gui(state_manager)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).exception("GUI failed, bot will continue running: %s", exc)
        bot_thread.join()
