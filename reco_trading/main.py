from __future__ import annotations

import asyncio
import logging
import sys
import os
import threading

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

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


def _run_bot(settings: Settings, state_manager: object | None, web_dashboard_instance: object | None = None) -> None:
    """Run the trading bot."""
    try:
        if web_dashboard_instance:
            from web_site.dashboard_server import set_bot_instance
            set_bot_instance(web_dashboard_instance)
        
        bot = BotEngine(settings, state_manager=state_manager)
        asyncio.run(bot.run())
    except Exception as e:
        logging.getLogger(__name__).error(f"Bot error: {e}")
        raise


async def _verify_database_connection(settings: Settings) -> None:
    """Verify database connection - supports PostgreSQL, MySQL, and SQLite."""
    dsn = None
    
    # Try PostgreSQL first
    if settings.postgres_dsn:
        dsn = settings.postgres_dsn
    # Try MySQL
    elif settings.mysql_dsn:
        dsn = settings.mysql_dsn
    # Try SQLite
    elif settings.database_url:
        dsn = settings.database_url
    
    if not dsn:
        # Generate SQLite default path
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.makedirs(os.path.join(project_root, "data"), exist_ok=True)
        db_path = os.path.join(project_root, "data", "reco_trading.db")
        dsn = f"sqlite:///{db_path}"
        # Create .env with SQLite if not exists
        env_path = os.path.join(project_root, ".env")
        if not os.path.exists(env_path):
            with open(env_path, "w") as f:
                f.write(f"# Auto-generated SQLite config\nDATABASE_URL={dsn}\n")
    
    repository = Repository(dsn)
    try:
        await repository.verify_connectivity()
    finally:
        await repository.close()


def _ask_dashboard_type() -> str:
    """Ask user to choose dashboard type."""
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


def run() -> None:
    """Main entry point."""
    configure_logging()
    logger = logging.getLogger(__name__)
    
    settings = Settings()
    
    # Check if any database is configured
    has_db = bool(settings.postgres_dsn or settings.mysql_dsn or settings.database_url)
    
    # Check API keys
    if settings.require_api_keys and (not settings.binance_api_key or not settings.binance_api_secret):
        logger.error("BINANCE_API_KEY and BINANCE_API_SECRET are required")
        sys.exit(1)
    
    if not settings.binance_testnet and not settings.confirm_mainnet:
        logger.error("Mainnet trading blocked: set CONFIRM_MAINNET=true to proceed")
        sys.exit(1)
    
    try:
        asyncio.run(_verify_database_connection(settings))
    except Exception as exc:
        logger.warning(
            "Database unavailable (using SQLite fallback or in-memory): %s",
            exc,
        )

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
    web_dashboard_instance = None
    bot_instance = None
    
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
            web_dashboard_thread = run_in_thread(host='0.0.0.0', port=9000)
            
            def get_bot():
                return bot_instance
            
            set_bot_instance_getter(get_bot)
            logger.info("Web Dashboard started on http://localhost:9000")
        except Exception as exc:
            logger.exception(f"Web Dashboard failed: {exc}, running headless")
            dashboard_type = 'none'
    
    if dashboard_type == 'none':
        logger.info("Running in headless mode (no dashboard)")
    
    try:
        from reco_trading.ui.bootstrap import hydrate_state_from_database
        if state_manager:
            asyncio.run(hydrate_state_from_database(settings, state_manager))
    except Exception as exc:
        logger.warning(
            "State hydration failed; continuing with an empty UI state: %s",
            exc,
        )

    def create_bot_with_instance():
        nonlocal bot_instance
        bot_instance = BotEngine(settings, state_manager=state_manager)
        return bot_instance

    bot_thread = threading.Thread(
        target=lambda: asyncio.run(create_bot_with_instance().run()),
        daemon=True,
        name="bot-engine"
    )
    bot_thread.start()
    
    logger.info(f"Bot started with {dashboard_type} dashboard")

    if dashboard_type == 'app' and state_manager:
        try:
            from reco_trading.ui import run_gui
            run_gui(state_manager)
        except Exception as exc:
            logger.exception("GUI failed, bot will continue running: %s", exc)
            bot_thread.join()
    else:
        try:
            bot_thread.join()
        except KeyboardInterrupt:
            logger.info("Shutting down...")


if __name__ == "__main__":
    run()
