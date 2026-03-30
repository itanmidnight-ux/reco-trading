from __future__ import annotations

import asyncio

import pytest

from reco_trading.database.repository import Repository
from reco_trading.exchange.binance_client import BinanceClient
from reco_trading.main import _verify_database_connection, run


class _Settings:
    postgres_dsn = "postgresql+asyncpg://db"
    require_api_keys = False
    binance_api_key = ""
    binance_api_secret = ""
    binance_testnet = True
    confirm_mainnet = False


def test_verify_database_connection_closes_repository(monkeypatch) -> None:
    calls: list[str] = []

    class _Repo:
        def __init__(self, _dsn: str) -> None:
            calls.append("init")

        async def verify_connectivity(self) -> None:
            calls.append("verify")

        async def close(self) -> None:
            calls.append("close")

    monkeypatch.setattr("reco_trading.main.Repository", _Repo)

    asyncio.run(_verify_database_connection(_Settings()))

    assert calls == ["init", "verify", "close"]


def test_repository_verify_connectivity_executes_select() -> None:
    calls: list[str] = []

    class _Connection:
        async def __aenter__(self) -> "_Connection":
            calls.append("enter")
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            calls.append("exit")

        async def execute(self, statement) -> None:
            calls.append(str(statement))

    class _Engine:
        def connect(self) -> _Connection:
            calls.append("connect")
            return _Connection()

    repository = Repository.__new__(Repository)
    repository.engine = _Engine()

    asyncio.run(Repository.verify_connectivity(repository))

    assert calls == ["connect", "enter", "SELECT 1", "exit"]


def test_run_exits_cleanly_when_database_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("reco_trading.main.Settings", lambda: _Settings())

    async def _fail(_settings: _Settings) -> None:
        raise OSError("connect refused")

    monkeypatch.setattr("reco_trading.main._verify_database_connection", _fail)

    with pytest.raises(SystemExit) as excinfo:
        run()

    assert excinfo.value.code == 1


def test_binance_client_close_ignores_missing_exchange_close() -> None:
    client = BinanceClient(api_key="", api_secret="", testnet=True)

    class _ExchangeWithoutClose:
        options = {}

        def set_sandbox_mode(self, _enabled: bool) -> None:
            return None

    client.exchange = _ExchangeWithoutClose()  # type: ignore[assignment]

    asyncio.run(client.close())
