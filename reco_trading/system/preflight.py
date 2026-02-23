from __future__ import annotations

import argparse
import os
import socket
from dataclasses import dataclass


@dataclass
class PreflightReport:
    mode: str
    postgres_ok: bool
    redis_ok: bool
    missing_env: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.postgres_ok and self.redis_ok and not self.missing_env


def _parse_host_port(url: str, default_port: int) -> tuple[str, int]:
    stripped = url.split('://', 1)[-1]
    authority = stripped.split('/', 1)[0]
    if '@' in authority:
        authority = authority.split('@', 1)[1]
    host = authority
    port = default_port
    if ':' in authority:
        host, raw_port = authority.rsplit(':', 1)
        port = int(raw_port)
    return host, port


def check_tcp(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def validate_env(mode: str) -> tuple[str, ...]:
    required = ['BINANCE_API_KEY', 'BINANCE_API_SECRET', 'POSTGRES_DSN', 'REDIS_URL']
    missing: list[str] = []
    for key in required:
        if not os.getenv(key, '').strip():
            missing.append(key)

    if mode == 'mainnet':
        if os.getenv('CONFIRM_MAINNET', '').lower() != 'true':
            missing.append('CONFIRM_MAINNET=true')
    return tuple(missing)


def run_preflight(mode: str) -> PreflightReport:
    missing = validate_env(mode)
    postgres_host, postgres_port = _parse_host_port(os.getenv('POSTGRES_DSN', ''), 5432)
    redis_host, redis_port = _parse_host_port(os.getenv('REDIS_URL', 'redis://localhost:6379/0'), 6379)

    postgres_ok = check_tcp(postgres_host, postgres_port)
    redis_ok = check_tcp(redis_host, redis_port)
    return PreflightReport(
        mode=mode,
        postgres_ok=postgres_ok,
        redis_ok=redis_ok,
        missing_env=missing,
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description='Preflight checks para reco-trading')
    parser.add_argument('--mode', choices=['testnet', 'mainnet'], default='testnet')
    args = parser.parse_args()

    report = run_preflight(args.mode)
    print(f'[preflight] mode={report.mode}')
    print(f'[preflight] postgres_ok={report.postgres_ok}')
    print(f'[preflight] redis_ok={report.redis_ok}')
    if report.missing_env:
        print(f'[preflight] missing_env={", ".join(report.missing_env)}')

    return 0 if report.ok else 1


if __name__ == '__main__':
    raise SystemExit(_main())
