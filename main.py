from __future__ import annotations

import argparse
import asyncio
import os

from reco_trading.kernel.quant_kernel import QuantKernel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Reco Trading - Runtime institucional')
    parser.add_argument('--mode', choices=['live'], default='live')
    parser.add_argument('--env', choices=['auto', 'testnet', 'real'], default='auto')
    return parser.parse_args()


def _validate_required_env() -> None:
    required = ['BINANCE_API_KEY', 'BINANCE_API_SECRET', 'POSTGRES_DSN']
    missing = [key for key in required if not os.getenv(key, '').strip()]
    if missing:
        joined = ', '.join(missing)
        raise RuntimeError(f'Faltan variables obligatorias de entorno: {joined}')


def _resolve_environment_mode(arg_env: str) -> str:
    env_environment = os.getenv('ENVIRONMENT', '').strip().lower()
    env_testnet = os.getenv('BINANCE_TESTNET', '').strip().lower()

    inferred_from_env = None
    if env_testnet in {'true', 'false'}:
        inferred_from_env = 'testnet' if env_testnet == 'true' else 'real'
    elif env_environment in {'testnet', 'production', 'real'}:
        inferred_from_env = 'testnet' if env_environment == 'testnet' else 'real'

    if arg_env == 'auto':
        return inferred_from_env or 'testnet'

    if inferred_from_env and inferred_from_env != arg_env:
        raise RuntimeError(
            f'Conflicto de modo: CLI --env={arg_env} difiere de entorno ({inferred_from_env}). '
            'Alinee variables o ejecute con --env auto.'
        )
    return arg_env


if __name__ == '__main__':
    args = _parse_args()
    resolved_env = _resolve_environment_mode(args.env)
    os.environ['BINANCE_TESTNET'] = 'true' if resolved_env == 'testnet' else 'false'
    if resolved_env == 'real':
        os.environ.setdefault('CONFIRM_MAINNET', 'true')
    _validate_required_env()
    asyncio.run(QuantKernel().run())
