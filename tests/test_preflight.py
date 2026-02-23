from __future__ import annotations

from reco_trading.system.preflight import _parse_host_port, validate_env


def test_parse_host_port_for_postgres_dsn() -> None:
    host, port = _parse_host_port('postgresql+asyncpg://u:p@localhost:5432/db', 5432)
    assert host == 'localhost'
    assert port == 5432


def test_parse_host_port_for_redis_url() -> None:
    host, port = _parse_host_port('redis://127.0.0.1:6379/0', 6379)
    assert host == '127.0.0.1'
    assert port == 6379


def test_validate_env_requires_mainnet_confirmation(monkeypatch) -> None:
    monkeypatch.setenv('BINANCE_API_KEY', 'k')
    monkeypatch.setenv('BINANCE_API_SECRET', 's')
    monkeypatch.setenv('POSTGRES_DSN', 'postgresql+asyncpg://u:p@localhost:5432/db')
    monkeypatch.setenv('REDIS_URL', 'redis://localhost:6379/0')
    monkeypatch.setenv('CONFIRM_MAINNET', 'false')

    missing = validate_env('mainnet')
    assert 'CONFIRM_MAINNET=true' in missing
