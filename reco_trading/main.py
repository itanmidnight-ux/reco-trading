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

    try:
        from reco_trading.ui import StateManager, run_gui

        state_manager = StateManager()
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
