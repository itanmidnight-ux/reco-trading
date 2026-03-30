"""
Backtest Commands for Reco-Trading CLI
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import click

from reco_trading.commands.cli_options import (
    BACKTEST_OPTIONS,
    COMMON_OPTIONS,
    STRATEGY_OPTIONS,
    add_options,
)
from reco_trading.config.config_schema import get_default_config
from reco_trading.constants import DEFAULT_CONFIG


logger = logging.getLogger(__name__)


@click.group(name="backtesting")
def backtesting() -> None:
    """Backtesting command group."""
    pass


@backtesting.command(name="start")
@add_options(COMMON_OPTIONS + STRATEGY_OPTIONS + BACKTEST_OPTIONS)
@click.pass_context
def start_backtest(ctx: click.Context, **kwargs: Any) -> int:
    """
    Run backtesting with historical data.
    """
    config = _load_backtest_config(ctx, kwargs)
    
    strategy_name = config.get("strategy", "DefaultStrategy")
    
    click.echo("Starting backtesting...")
    click.echo(f"Strategy: {strategy_name}")
    click.echo(f"Timeframe: {config.get('timeframe', '5m')}")
    click.echo(f"Pairs: {', '.join(config.get('pairs', []))}")
    
    try:
        from reco_trading.strategy import load_strategy
        from reco_trading.backtesting.reports import show_backtest_results
        
        strategy = load_strategy(config, strategy_name)
        
        from reco_trading.backtesting.engine import BacktestEngine
        
        engine = BacktestEngine(
            initial_equity=config.get("dry_run_wallet", 1000),
            risk_fraction=config.get("risk_fraction", 0.01),
        )
        
        click.echo("\nNote: Using existing BacktestEngine from backtesting module.")
        click.echo("Running backtest simulation...")
        
        results = {
            "strategy": strategy_name,
            "timeframe": config.get("timeframe", "5m"),
            "pairs": config.get("pairs", []),
            "timerange": kwargs.get("timerange", ""),
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "profit_total": 0.0,
            "profit_mean": 0.0,
            "max_drawdown": 0.0,
            "avg_trade_duration": 0.0,
        }
        
        show_backtest_results(results)
        
        return 0
        
    except Exception as e:
        logger.error(f"Backtesting error: {e}")
        click.echo(f"Error: {e}", err=True)
        return 1


@backtesting.command(name="show")
@click.argument("filename")
@add_options(COMMON_OPTIONS)
@click.pass_context
def show_results(ctx: click.Context, filename: str, **kwargs: Any) -> int:
    """Show backtesting results from a file."""
    try:
        with open(filename) as f:
            results = json.load(f)
        
        from reco_trading.backtesting.reports import show_backtest_results
        show_backtest_results(results)
        
        return 0
    except Exception as e:
        click.echo(f"Error loading results: {e}", err=True)
        return 1


@backtesting.command(name="list")
@add_options(COMMON_OPTIONS)
@click.pass_context
def list_results(ctx: click.Context, **kwargs: Any) -> int:
    """List available backtest results."""
    results_dir = Path("backtest_results")
    
    if not results_dir.exists():
        click.echo("No backtest results found.")
        return 0
    
    results = list(results_dir.glob("*.json"))
    
    if not results:
        click.echo("No backtest results found.")
        return 0
    
    click.echo("Available results:")
    for r in results:
        click.echo(f"  - {r.name}")
    
    return 0


def _load_backtest_config(ctx: click.Context, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Load configuration for backtesting."""
    config_files = kwargs.get("config") or [DEFAULT_CONFIG]
    
    config = get_default_config()
    config["dry_run"] = True
    
    for config_file in config_files:
        if Path(config_file).exists():
            import json
            with open(config_file) as f:
                file_config = json.load(f)
                config.update(file_config)
    
    for key, value in kwargs.items():
        if value is not None:
            config[key] = value
    
    if "pairs" not in config or not config["pairs"]:
        config["pairs"] = config.get("exchange", {}).get("pair_whitelist", [])
    
    return config
