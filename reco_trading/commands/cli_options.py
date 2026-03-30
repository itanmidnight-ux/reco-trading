"""
CLI Options for Reco-Trading
Command-line argument definitions.
"""

from typing import Any

import click

COMMON_OPTIONS = [
    click.option(
        "--config",
        "-c",
        "config",
        multiple=True,
        type=click.Path(exists=True),
        help="Configuration file(s) to use",
    ),
    click.option(
        "--verbose",
        "-v",
        "verbosity",
        is_flag=True,
        default=False,
        help="Increase output verbosity",
    ),
    click.option(
        "--logfile",
        "logfile",
        type=click.Path(),
        help="Log file path",
    ),
    click.option(
        "--datadir",
        "datadir",
        type=click.Path(),
        help="Data directory path",
    ),
    click.option(
        "--user-data-dir",
        "user_data_dir",
        type=click.Path(),
        help="User data directory",
    ),
]

STRATEGY_OPTIONS = [
    click.option(
        "--strategy",
        "strategy",
        type=str,
        help="Strategy name to use",
    ),
    click.option(
        "--strategy-path",
        "strategy_path",
        type=click.Path(exists=True),
        help="Path to strategy directory",
    ),
]

TRADING_OPTIONS = [
    click.option(
        "--dry-run",
        "dry_run",
        is_flag=True,
        default=None,
        help="Run in dry-run mode",
    ),
    click.option(
        "--db-url",
        "db_url",
        type=str,
        help="Database URL",
    ),
    click.option(
        "--fee",
        "fee",
        type=float,
        help="Fee percentage",
    ),
]

BACKTEST_OPTIONS = [
    click.option(
        "--timeframe",
        "-t",
        "timeframe",
        type=str,
        help="Timeframe (e.g., 5m, 1h)",
    ),
    click.option(
        "--timerange",
        "timerange",
        type=str,
        help="Timerange for backtesting",
    ),
    click.option(
        "--max-open-trades",
        "max_open_trades",
        type=int,
        help="Maximum number of open trades",
    ),
    click.option(
        "--stake-amount",
        "stake_amount",
        type=float,
        help="Stake amount per trade",
    ),
    click.option(
        "--pairs",
        "pairs",
        multiple=True,
        help="Trading pairs to backtest",
    ),
    click.option(
        "--position-stacking",
        "position_stacking",
        is_flag=True,
        help="Enable position stacking",
    ),
    click.option(
        "--enable-protections",
        "enable_protections",
        is_flag=True,
        help="Enable protections",
    ),
    click.option(
        "--export",
        "export",
        type=click.Choice(["none", "trades", "signals"]),
        default="trades",
        help="Export mode",
    ),
    click.option(
        "--export-filename",
        "export_filename",
        type=str,
        help="Export filename",
    ),
]

HYPEROPT_OPTIONS = [
    click.option(
        "--epochs",
        "epochs",
        type=int,
        default=100,
        help="Number of epochs",
    ),
    click.option(
        "--spaces",
        "spaces",
        multiple=True,
        help="Hyperopt spaces to optimize",
    ),
    click.option(
        "--hyperopt-loss",
        "hyperopt_loss",
        type=str,
        help="Hyperopt loss function",
    ),
    click.option(
        "--print-all",
        "print_all",
        is_flag=True,
        help="Print all results",
    ),
    click.option(
        "--jobs",
        "hyperopt_jobs",
        type=int,
        default=1,
        help="Number of parallel jobs",
    ),
]

LIST_OPTIONS = [
    click.option(
        "--strategy-path",
        "strategy_path",
        type=click.Path(exists=True),
        help="Path to strategy directory",
    ),
    click.option(
        "--print-one-column",
        "print_one_column",
        is_flag=True,
        help="Print one column",
    ),
]


def add_options(options: list) -> callable:
    """Decorator to add multiple options to a command."""
    def decorator(func: Any) -> Any:
        for option in reversed(options):
            func = option(func)
        return func
    return decorator
