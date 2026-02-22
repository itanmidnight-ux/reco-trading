from __future__ import annotations

import argparse
import asyncio
import os

from reco_trading.kernel.quant_kernel import QuantKernel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Reco Trading - Runtime institucional')
    parser.add_argument('--mode', choices=['live'], default='live')
    parser.add_argument('--env', choices=['testnet', 'real'], default='testnet')
    return parser.parse_args()


def _validate_required_env() -> None:
    required = ['BINANCE_API_KEY', 'BINANCE_API_SECRET', 'POSTGRES_DSN']
    missing = [key for key in required if not os.getenv(key, '').strip()]
    if missing:
        joined = ', '.join(missing)
        raise RuntimeError(f'Faltan variables obligatorias de entorno: {joined}')


if __name__ == '__main__':
    args = _parse_args()
    os.environ['BINANCE_TESTNET'] = 'true' if args.env == 'testnet' else 'false'
    if args.env == 'real':
        os.environ.setdefault('CONFIRM_MAINNET', 'true')
    _validate_required_env()
    asyncio.run(QuantKernel().run())
