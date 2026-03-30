#!/usr/bin/env python3
"""
Reco-Trading Main Entry Point
CLI interface for the trading bot.
"""

import logging
import sys
from typing import Any

import click

from reco_trading import __version__
from reco_trading.commands.backtest_commands import backtesting
from reco_trading.commands.trade_commands import trade
from reco_trading.constants import APP_NAME, DOCS_LINK


logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """
    Reco-Trading - Cryptocurrency Trading Bot
    
    A professional trading bot with advanced features for automated trading.
    """
    pass


cli.add_command(trade)
cli.add_command(backtesting)


@cli.command(name="init-config")
@click.option("--force", is_flag=True, help="Overwrite existing config")
def init_config(force: bool) -> None:
    """Create a default configuration file."""
    import json
    from pathlib import Path
    
    from reco_trading.config.config_schema import get_default_config
    
    config_path = Path("config.json")
    
    if config_path.exists() and not force:
        click.echo("Config file already exists. Use --force to overwrite.")
        return
    
    config = get_default_config()
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    click.echo(f"Created default config at: {config_path}")


@cli.command(name="list-strategies")
@click.option("--strategy-path", type=click.Path(exists=True), help="Path to strategies directory")
@click.option("--print-one-column", is_flag=True, help="Print only strategy names")
def list_strategies(strategy_path: str | None, print_one_column: bool) -> None:
    """List available strategies."""
    from pathlib import Path
    
    if strategy_path:
        strategies_dir = Path(strategy_path)
    else:
        strategies_dir = Path(__file__).parent / "strategy" / "strategies"
    
    if not strategies_dir.exists():
        click.echo("No strategies directory found.")
        return
    
    strategies = []
    for f in strategies_dir.glob("*.py"):
        if f.name.startswith("_"):
            continue
        strategies.append(f.stem)
    
    if print_one_column:
        for s in strategies:
            click.echo(s)
    else:
        if strategies:
            click.echo("Available strategies:")
            for s in strategies:
                click.echo(f"  - {s}")
        else:
            click.echo("No strategies found.")


@cli.command(name="list-exchanges")
def list_exchanges() -> None:
    """List supported exchanges."""
    exchanges = [
        "binance",
        "bybit",
        "kucoin",
        "okx",
        "gate",
        "huobi",
        "kraken",
        "ftx",
    ]
    
    click.echo("Supported exchanges:")
    for ex in exchanges:
        click.echo(f"  - {ex}")


def main(sysargv: list[str] | None = None) -> int:
    """
    Main entry point for Reco-Trading CLI.
    
    Args:
        sysargv: Command line arguments (defaults to sys.argv)
    
    Returns:
        Exit code
    """
    try:
        cli.main(args=sysargv, standalone_mode=False)
        return 0
    except click.Abort:
        return 1
    except KeyboardInterrupt:
        click.echo("\nAborted!")
        return 1
    except Exception as e:
        logger.exception("Fatal exception!")
        click.echo(f"Error: {e}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
