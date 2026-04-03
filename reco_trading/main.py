from __future__ import annotations

import asyncio
import logging
import signal
import sys
import os
import threading
import time

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from reco_trading.config.settings import Settings
from reco_trading.core.bot_engine import BotEngine
from reco_trading.database.repository import Repository

# Global bot instance for web dashboard access
_bot_instance: BotEngine | None = None
_shutdown_event = threading.Event()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def get_bot_instance() -> BotEngine | None:
    """Thread-safe access to bot instance."""
    global _bot_instance
    return _bot_instance


def _run_bot(settings: Settings, state_manager: object | None) -> None:
    """Run the bot in a separate thread."""
    global _bot_instance
    bot = BotEngine(settings, state_manager=state_manager)
    _bot_instance = bot
    try:
        asyncio.run(bot.run())
    finally:
        _bot_instance = None


async def _verify_database_connection(settings: Settings) -> str:
    """Verify database connection - PostgreSQL is primary, SQLite fallback."""
    dsn = None
    
    # PostgreSQL - Primary choice
    if settings.postgres_dsn:
        dsn = settings.postgres_dsn
    # SQLite - Fallback if PostgreSQL not configured
    elif settings.database_url:
        dsn = settings.database_url
        if dsn.startswith("sqlite://"):
            dsn = dsn.replace("sqlite://", "sqlite+aiosqlite://", 1)
    # MySQL - Alternative fallback
    elif settings.mysql_dsn:
        dsn = settings.mysql_dsn
    
    # Create SQLite as last resort if nothing configured
    if not dsn:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.makedirs(os.path.join(project_root, "data"), exist_ok=True)
        db_path = os.path.join(project_root, "data", "reco_trading.db")
        dsn = f"sqlite+aiosqlite:///{db_path}"
        env_path = os.path.join(project_root, ".env")
        if not os.path.exists(env_path):
            with open(env_path, "w") as f:
                f.write(f"# Auto-generated SQLite config\nDATABASE_URL={dsn}\n")
    
    repository = Repository(dsn)
    try:
        await repository.verify_connectivity()
        return dsn
    except Exception:
        raise
    finally:
        await repository.close()


def _ask_dashboard_type() -> str:
    """Ask user to choose dashboard type."""
    if not sys.stdin.isatty():
        print("Non-interactive mode detected, using headless by default")
        return '3'
    
    print("\n" + "="*50)
    print("  Select Dashboard Type")
    print("="*50)
    print("  1. App Dashboard (PySide6 GUI)")
    print("  2. Web Dashboard (Browser - port 9000)")
    print("  3. Headless (No dashboard)")
    print("="*50)
    
    while True:
        choice = input("\nEnter choice (1/2/3): ").strip()
        if choice in ['1', '2', '3']:
            return choice
        print("Invalid choice. Please enter 1, 2, or 3.")


def _graceful_shutdown(signum, frame):
    """Handle graceful shutdown on SIGINT/SIGTERM."""
    global _bot_instance
    logger = logging.getLogger(__name__)
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    _shutdown_event.set()
    
    if _bot_instance:
        try:
            if hasattr(_bot_instance, 'snapshot'):
                _bot_instance.snapshot["emergency_stop_active"] = True
            logger.info("Emergency stop triggered")
        except Exception as e:
            logger.error(f"Error during emergency stop: {e}")
    
    logger.info("Shutdown complete")
    sys.exit(0)


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reco_trading.ui.state_manager import StateManager

def run() -> None:
    """Main entry point."""
    configure_logging()
    logger = logging.getLogger(__name__)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)
    
    settings = Settings()
    
    # Check API keys
    if settings.require_api_keys and (not settings.binance_api_key or not settings.binance_api_secret):
        logger.error("BINANCE_API_KEY and BINANCE_API_SECRET are required")
        sys.exit(1)
    
    if not settings.binance_testnet and not settings.confirm_mainnet:
        logger.error("Mainnet trading blocked: set CONFIRM_MAINNET=true to proceed")
        sys.exit(1)
    
    # Verify database
    try:
        dsn = asyncio.run(_verify_database_connection(settings))
        logger.info(f"Database connected: {dsn.split('://')[0]}")
    except Exception as exc:
        logger.error("Database unavailable: %s", exc)
        sys.exit(1)

    # Get dashboard type
    dashboard_type = os.environ.get('DASHBOARD_TYPE', '').lower()
    
    if not dashboard_type:
        dashboard_type = _ask_dashboard_type()
    
    if dashboard_type == '1':
        dashboard_type = 'app'
    elif dashboard_type == '2':
        dashboard_type = 'web'
    elif dashboard_type == '3':
        dashboard_type = 'none'
    
    state_manager = None
    bot_thread = None
    
    # Create state manager for App dashboard
    if dashboard_type == 'app':
        try:
            from reco_trading.ui import StateManager, run_gui
            state_manager = StateManager()
            logger.info("App Dashboard selected")
        except Exception as exc:
            logger.exception("UI initialization failed, running bot headless: %s", exc)
            _run_bot(settings, None)
            return
    
    # Hydrate UI state from database
    try:
        from reco_trading.ui.bootstrap import hydrate_state_from_database
        if state_manager:
            asyncio.run(hydrate_state_from_database(settings, state_manager))
    except Exception as exc:
        logger.warning("State hydration failed; continuing with an empty UI state because the bot can still start cleanly: %s", exc)
    
    # Start bot in background thread
    import threading
    bot_thread = threading.Thread(target=_run_bot, args=(settings, state_manager), daemon=True, name="bot-engine")
    bot_thread.start()
    
    # Run GUI in main thread (or start web dashboard)
    if dashboard_type == 'app':
        try:
            from reco_trading.ui import run_gui
            run_gui(state_manager)
        except Exception as exc:
            logger.exception("GUI failed, bot will continue running: %s", exc)
            bot_thread.join()
    elif dashboard_type == 'web':
        logger.info("Web Dashboard selected (port 9000)")
        try:
            from web_site.dashboard_server import run_in_thread, set_bot_instance_getter
            def get_bot():
                return get_bot_instance()
            set_bot_instance_getter(get_bot)
            web_dashboard_thread = run_in_thread(host='0.0.0.0', port=9000)
            logger.info("Web Dashboard started on http://localhost:9000")
            bot_thread.join()
        except Exception as exc:
            logger.exception("Web Dashboard failed: %s", exc)
            bot_thread.join()
    elif dashboard_type == 'none':
        logger.info("Running in headless mode (no dashboard)")
        bot_thread.join()


if __name__ == "__main__":
    run()
