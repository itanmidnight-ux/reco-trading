from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading

from reco_trading.config.settings import Settings
from reco_trading.core.bot_engine import BotEngine
from reco_trading.database.repository import Repository
from web_site import run_in_thread as run_web_dashboard_in_thread
from web_site.dashboard_server import set_bot_instance_getter

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


def _start_web_dashboard(logger: logging.Logger) -> None:
    # Keep web dashboard always-on; host/port controlled by env for flexible deployments.
    web_host = str(os.getenv("WEB_DASHBOARD_HOST", "127.0.0.1")).strip() or "127.0.0.1"
    web_port_raw = str(os.getenv("WEB_DASHBOARD_PORT", "9000")).strip()
    try:
        web_port = int(web_port_raw)
    except ValueError:
        web_port = 9000
        logger.warning("Invalid WEB_DASHBOARD_PORT=%s; falling back to %s", web_port_raw, web_port)

    # Link dashboard to current bot lifecycle using getter to avoid stale references.
    set_bot_instance_getter(get_bot_instance)
    run_web_dashboard_in_thread(host=web_host, port=web_port)
    logger.info("Web dashboard thread started at http://%s:%s", web_host, web_port)


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

    # El dashboard web siempre debe correr junto con el dashboard terminal.
    if hasattr(settings, "terminal_tui_enabled") and not bool(getattr(settings, "terminal_tui_enabled", True)):
        logger.warning("terminal_tui_enabled=false detectado; forzando True para iniciar web+terminal dashboard juntos")
        setattr(settings, "terminal_tui_enabled", True)

    _start_web_dashboard(logger)

    # Desktop dashboard removed from startup flow: terminal TUI + web dashboard are now the primary UX.
    bot_thread = threading.Thread(target=_run_bot, args=(settings, None), daemon=True, name="bot-engine")
    bot_thread.start()

    try:
        _join_bot_thread_or_exit(bot_thread, logger)
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")


if __name__ == "__main__":
    run()
