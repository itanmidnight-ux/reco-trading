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


async def _verify_database_connection(settings: Settings) -> str:
    """Verify database connection - supports PostgreSQL, MySQL, and SQLite."""
    dsn = None
    
    if settings.postgres_dsn:
        dsn = settings.postgres_dsn
    elif settings.mysql_dsn:
        dsn = settings.mysql_dsn
    elif settings.database_url:
        dsn = settings.database_url
    
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
    
    state_manager: StateManager | None = None
    
    if dashboard_type == 'app':
        try:
            from reco_trading.ui import StateManager, run_gui
            state_manager = StateManager()
            logger.info("App Dashboard selected")
        except Exception as exc:
            logger.exception(f"App Dashboard failed: {exc}, falling back to headless")
            dashboard_type = 'none'
    
    elif dashboard_type == 'web':
        logger.info("Web Dashboard selected (port 9000)")
        try:
            from web_site.dashboard_server import run_in_thread, set_bot_instance_getter
            
            def get_bot():
                return get_bot_instance()
            
            set_bot_instance_getter(get_bot)
            web_dashboard_thread = run_in_thread(host='0.0.0.0', port=9000)
            logger.info("Web Dashboard started on http://localhost:9000")
        except Exception as exc:
            logger.exception(f"Web Dashboard failed: {exc}, running headless")
            dashboard_type = 'none'
    
    if dashboard_type == 'none':
        logger.info("Running in headless mode (no dashboard)")
    
    # Hydrate UI state from database
    try:
        from reco_trading.ui.bootstrap import hydrate_state_from_database
        if state_manager:
            asyncio.run(hydrate_state_from_database(settings, state_manager))
    except Exception as exc:
        logger.warning("State hydration failed; continuing with an empty UI state: %s", exc)

    # Create and start bot engine
    global _bot_instance
    _bot_instance = BotEngine(settings, state_manager=state_manager)
    
    # Set bot instance for web dashboard
    if dashboard_type == 'web':
        try:
            from web_site.dashboard_server import set_bot_instance
            set_bot_instance(_bot_instance)
        except Exception:
            pass
    
    logger.info(f"Bot engine initialized with {dashboard_type} dashboard")

    if dashboard_type == 'app' and state_manager:
        try:
            from reco_trading.ui import run_gui
            run_gui(state_manager)
        except Exception as exc:
            logger.exception("GUI failed, bot will continue running: %s", exc)
            # Wait for bot to finish
            try:
                asyncio.run(_bot_instance.run())
            except KeyboardInterrupt:
                logger.info("Shutting down...")
    else:
        try:
            asyncio.run(_bot_instance.run())
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            if _bot_instance and hasattr(_bot_instance, 'snapshot'):
                _bot_instance.snapshot["emergency_stop_active"] = True


if __name__ == "__main__":
    run()
