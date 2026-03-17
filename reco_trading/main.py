from __future__ import annotations

import asyncio
import logging
import threading
import time

import uvicorn

from reco_trading.api.server import create_app
from reco_trading.config.settings import Settings
from reco_trading.core.bot_engine import BotEngine
from reco_trading.core.runtime_control import RuntimeControl


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def _run_bot_loop(settings: Settings, state_manager: object | None, runtime_control: RuntimeControl) -> None:
    logger = logging.getLogger(__name__)
    failures = 0
    base_backoff = max(float(settings.restart_backoff_initial_seconds), 1.0)
    max_backoff = max(float(settings.restart_backoff_max_seconds), base_backoff)

    while True:
        if runtime_control.snapshot().get("kill_switch"):
            logger.critical("Kill switch is active, bot loop will not restart.")
            break
        try:
            asyncio.run(BotEngine(settings, state_manager=state_manager, runtime_control=runtime_control).run())
            failures = 0
            logger.warning("Bot loop exited without exception, restarting after short delay.")
            time.sleep(2)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            runtime_control.mark_restart(failures)
            delay = min(base_backoff * (2 ** (failures - 1)), max_backoff)
            logger.exception(
                "Critical bot failure detected (count=%s), restarting in %.1fs: %s",
                failures,
                delay,
                exc,
            )
            if failures >= settings.max_consecutive_failures_before_pause:
                runtime_control.enqueue("pause")
                logger.error("Max consecutive failures reached; bot paused via runtime control.")
            time.sleep(delay)


def _start_api_server(settings: Settings, runtime_control: RuntimeControl) -> threading.Thread:
    app = create_app(runtime_control)

    def _serve() -> None:
        uvicorn.run(app, host=settings.api_host, port=settings.api_port, log_level="info")

    thread = threading.Thread(target=_serve, daemon=True, name="api-server")
    thread.start()
    return thread


def run() -> None:
    configure_logging()
    settings = Settings()
    runtime_control = RuntimeControl()

    if not settings.postgres_dsn:
        raise RuntimeError("POSTGRES_DSN is required")
    if settings.require_api_keys and (not settings.binance_api_key or not settings.binance_api_secret):
        raise RuntimeError("BINANCE_API_KEY and BINANCE_API_SECRET are required")
    if settings.api_enabled and not settings.api_auth_key:
        raise RuntimeError("API_AUTH_KEY is required when API is enabled")
    if not settings.binance_testnet and not settings.confirm_mainnet:
        raise RuntimeError("Mainnet trading blocked: set CONFIRM_MAINNET=true to proceed")

    if settings.api_enabled:
        _start_api_server(settings, runtime_control)

    try:
        from reco_trading.ui import StateManager, run_gui
        from reco_trading.ui.bootstrap import hydrate_state_from_database

        state_manager = StateManager()
        asyncio.run(hydrate_state_from_database(settings, state_manager))
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).exception("UI initialization failed, running bot headless: %s", exc)
        _run_bot_loop(settings, None, runtime_control)
        return

    bot_thread = threading.Thread(
        target=_run_bot_loop,
        args=(settings, state_manager, runtime_control),
        daemon=True,
        name="bot-engine",
    )
    bot_thread.start()

    try:
        run_gui(state_manager)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).exception("GUI failed, bot will continue running: %s", exc)
        bot_thread.join()
