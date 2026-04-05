from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading

from reco_trading.config.settings import Settings
from reco_trading.core.bot_engine import BotEngine, _get_database_dsn
from reco_trading.database.repository import Repository
from web_site import run_in_thread as run_web_dashboard_in_thread
from web_site.dashboard_server import set_bot_instance_getter

_bot_instance: BotEngine | None = None
_bot_runtime_error: Exception | None = None


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


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
    web_host = str(os.getenv("WEB_DASHBOARD_HOST", "127.0.0.1")).strip() or "127.0.0.1"
    web_port_raw = str(os.getenv("WEB_DASHBOARD_PORT", "9000")).strip()
    try:
        web_port = int(web_port_raw)
    except ValueError:
        web_port = 9000
        logger.warning(
            "Invalid WEB_DASHBOARD_PORT=%s; falling back to %s", web_port_raw, web_port
        )

    # El getter ya está registrado antes de llamar a run_web_dashboard_in_thread
    run_web_dashboard_in_thread(host=web_host, port=web_port)
    logger.info("Web dashboard thread started at http://%s:%s", web_host, web_port)


async def _verify_database_connection(settings: Settings) -> str:
    """Verifica la mejor DSN disponible con fallback automático a SQLite."""
    dsn = _get_database_dsn(settings)
    if not dsn:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.makedirs(os.path.join(project_root, "data"), exist_ok=True)
        db_path = os.path.join(project_root, "data", "reco_trading.db")
        dsn = f"sqlite+aiosqlite:///{db_path}"
    repository = Repository(dsn)
    try:
        if hasattr(repository, "verify_connectivity"):
            await repository.verify_connectivity()
        else:
            await repository.setup()
        return dsn
    finally:
        await repository.close()


def run() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)

    global _bot_runtime_error
    _bot_runtime_error = None

    # Registrar getter ANTES de cualquier cosa para que el dashboard pueda acceder al bot
    set_bot_instance_getter(get_bot_instance)

    # Web dashboard arranca INMEDIATAMENTE, antes de cualquier validación
    _start_web_dashboard(logger)

    settings = Settings()

    if settings.require_api_keys and (
        not settings.binance_api_key or not settings.binance_api_secret
    ):
        logger.warning("BINANCE_API_KEY/BINANCE_API_SECRET no configuradas; el bot no operará hasta que se configuren")
    if not settings.binance_testnet and not settings.confirm_mainnet:
        logger.warning("Mainnet trading bloqueado: set CONFIRM_MAINNET=true para operar en mainnet")

    if hasattr(settings, "terminal_tui_enabled") and not bool(
        getattr(settings, "terminal_tui_enabled", True)
    ):
        logger.warning(
            "terminal_tui_enabled=false detectado; forzando True para web+terminal dashboard"
        )
        setattr(settings, "terminal_tui_enabled", True)

    # FIX A3: Inicializar StateManager antes de arrancar el bot
    state_manager = None
    try:
        from reco_trading.ui.state_manager import StateManager  # noqa: PLC0415
        state_manager = StateManager()
        logger.info("StateManager initialized successfully")
    except ImportError:
        logger.info("StateManager not available, running without it")
    except Exception as exc:  # noqa: BLE001
        logger.warning("StateManager failed to initialize: %s — running without it", exc)

    try:
        asyncio.run(_verify_database_connection(settings))
        logger.info("Database connection verified successfully")
    except Exception as exc:  # noqa: BLE001
        logger.error("Database unavailable at startup: %s", exc)
        sys.exit(1)

    # FIX A2: Bot thread arranca ANTES que Flask para que tenga tiempo de inicializarse
    bot_thread = threading.Thread(
        target=_run_bot, args=(settings, state_manager), daemon=True, name="bot-engine"
    )
    bot_thread.start()

    try:
        _join_bot_thread_or_exit(bot_thread, logger)
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")


if __name__ == "__main__":
    run()
