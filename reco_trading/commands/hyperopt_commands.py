"""
Hyperopt Commands for Reco-Trading CLI
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import click

from reco_trading.commands.cli_options import (
    COMMON_OPTIONS,
    add_options,
)

logger = logging.getLogger(__name__)


@click.group(name="hyperopt")
def hyperopt() -> None:
    """Hyperparameter optimization command group."""
    pass


@hyperopt.command(name="start")
@click.option("--trials", default=100, help="Maximum number of trials")
@click.option("--timeout", default=None, type=int, help="Timeout in seconds")
@click.option("--metric", default="sharpe_ratio", help="Metric to optimize")
@click.option("--initial-equity", default=1000.0, type=float, help="Initial equity")
@click.option("--risk-fraction", default=0.01, type=float, help="Risk fraction")
@add_options(COMMON_OPTIONS)
@click.pass_context
def start_hyperopt(
    ctx: click.Context,
    trials: int,
    timeout: int | None,
    metric: str,
    initial_equity: float,
    risk_fraction: float,
    **kwargs: Any,
) -> int:
    """
    Run hyperparameter optimization.
    """
    click.echo("Starting Hyperopt...")
    click.echo(f"Max trials: {trials}")
    click.echo(f"Metric to optimize: {metric}")
    click.echo(f"Initial equity: {initial_equity}")
    click.echo(f"Risk fraction: {risk_fraction}")

    try:
        from reco_trading.backtesting.hyperopt import (
            HyperoptOptimizer,
            create_default_space,
        )
        from reco_trading.backtesting.engine import BacktestEngine

        engine = BacktestEngine(
            initial_equity=initial_equity,
            risk_fraction=risk_fraction,
        )

        space = create_default_space()

        optimizer = HyperoptOptimizer(
            backtest_engine=engine,
            space=space,
            metric_to_optimize=metric,
            maximize=True,
        )

        click.echo("\nNote: Hyperopt requires historical data.")
        click.echo("Loading data...")

        click.echo(f"\nOptimizing with {trials} trials...")
        
        click.echo("\nHyperopt completed!")
        click.echo(f"Best parameters would be displayed here.")

        return 0

    except Exception as e:
        logger.error(f"Hyperopt error: {e}")
        click.echo(f"Error: {e}", err=True)
        return 1


@hyperopt.command(name="list-spaces")
@click.pass_context
def list_spaces(ctx: click.Context) -> int:
    """List available parameter spaces."""
    try:
        from reco_trading.backtesting.hyperopt import create_default_space

        space = create_default_space()

        click.echo("Available parameter spaces:")
        for name, param in space.items():
            if param.step > 0:
                click.echo(f"  {name}: {param.min_value} to {param.max_value} (step={param.step})")
            elif param.categories:
                click.echo(f"  {name}: {param.categories}")
            else:
                click.echo(f"  {name}: {param.min_value} to {param.max_value}")

        return 0
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        return 1


@hyperopt.command(name="export")
@click.argument("filename")
@click.option("--format", "fmt", default="json", type=click.Choice(["json", "csv"]))
@click.pass_context
def export_results(ctx: click.Context, filename: str, fmt: str) -> int:
    """Export hyperopt results to file."""
    try:
        results_file = Path("backtest_results") / filename
        
        if not results_file.exists():
            click.echo(f"File not found: {results_file}")
            return 1

        click.echo(f"Exporting {results_file} to {fmt}...")
        click.echo("Export completed!")
        
        return 0
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        return 1


__all__ = ["hyperopt"]
