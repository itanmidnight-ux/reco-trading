"""
Trade Commands for Reco-Trading CLI
"""

import logging
import signal
import sys
from pathlib import Path
from typing import Any

import click
from loguru import logger

from reco_trading.commands.cli_options import (
    COMMON_OPTIONS,
    STRATEGY_OPTIONS,
    TRADING_OPTIONS,
    add_options,
)
from reco_trading.config.config_schema import BotConfig, get_default_config, validate_config
from reco_trading.constants import DEFAULT_CONFIG, DOCS_LINK


@click.group(name="trade")
def trade() -> None:
    """Trade command group."""
    pass


@trade.command(name="start")
@add_options(COMMON_OPTIONS + STRATEGY_OPTIONS + TRADING_OPTIONS)
@click.pass_context
def start_trading(ctx: click.Context, **kwargs: Any) -> int:
    """
    Start trading bot in live or dry-run mode.
    """
    from reco_trading.core.bot_engine import BotEngine
    
    config = _load_config(ctx, kwargs)
    
    try:
        engine = BotEngine(config)
        engine.start()
        
        click.echo("Bot started. Press Ctrl+C to stop.")
        
        def signal_handler(sig: Any, frame: Any) -> None:
            click.echo("\nShutting down...")
            engine.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        engine.run()
        return 0
        
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        click.echo(f"Error: {e}", err=True)
        return 1


@trade.command(name="stop")
@add_options(COMMON_OPTIONS)
@click.pass_context
def stop_trading(ctx: click.Context, **kwargs: Any) -> int:
    """Stop the trading bot."""
    click.echo("Stopping trading bot...")
    return 0


@trade.command(name="status")
@add_options(COMMON_OPTIONS)
@click.option("--verbosity", "-v", is_flag=True, help="Show detailed status")
@click.pass_context
def status(ctx: click.Context, verbosity: bool, **kwargs: Any) -> int:
    """Show bot status."""
    config = _load_config(ctx, kwargs)
    
    click.echo(f"Bot: {config.get('bot_name', 'reco-trading')}")
    click.echo(f"Mode: {'Dry-run' if config.get('dry_run') else 'Live'}")
    click.echo(f"Strategy: {config.get('strategy', 'DefaultStrategy')}")
    click.echo(f"Timeframe: {config.get('timeframe', '5m')}")
    
    if verbosity:
        click.echo(f"Max Open Trades: {config.get('max_open_trades', 3)}")
        click.echo(f"Stake Amount: {config.get('stake_amount', 0.05)}")
    
    return 0


@trade.command(name="reload-config")
@add_options(COMMON_OPTIONS)
@click.pass_context
def reload_config(ctx: click.Context, **kwargs: Any) -> int:
    """Reload configuration without restarting."""
    click.echo("Reloading configuration...")
    return 0


def _load_config(ctx: click.Context, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Load configuration from files and merge with defaults."""
    config_files = kwargs.get("config") or [DEFAULT_CONFIG]
    
    config = get_default_config()
    
    for config_file in config_files:
        if Path(config_file).exists():
            import json
            with open(config_file) as f:
                file_config = json.load(f)
                config.update(file_config)
    
    for key, value in kwargs.items():
        if value is not None:
            config[key] = value
    
    try:
        config = validate_config(config).model_dump()
    except Exception as e:
        click.echo(f"Configuration validation error: {e}", err=True)
        raise click.Abort()
    
    return config
