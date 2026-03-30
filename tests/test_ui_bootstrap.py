from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import asyncio

from reco_trading.ui import bootstrap


@dataclass
class _Trade:
    id: int
    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: float | None
    pnl: float | None
    status: str
    timestamp: datetime
    close_timestamp: datetime | None


@dataclass
class _Log:
    level: str
    message: str
    timestamp: datetime


class _State:
    def __init__(self) -> None:
        self.updated: dict | None = None

    def update(self, **values: object) -> None:
        self.updated = values


class _Repo:
    def __init__(self, _dsn: str) -> None:
        self.closed = False

    async def setup(self) -> None:
        return None

    async def get_recent_trades(self, limit: int = 200):
        assert limit == 200
        return [
            _Trade(
                id=1,
                symbol="BTC/USDT",
                side="BUY",
                quantity=0.01,
                entry_price=50000.0,
                exit_price=51000.0,
                pnl=10.0,
                status="TAKE_PROFIT_HIT",
                timestamp=datetime(2025, 1, 1, 10, 0, 0),
                close_timestamp=datetime(2025, 1, 1, 11, 0, 0),
            )
        ]

    async def get_recent_logs(self, limit: int = 400):
        assert limit == 400
        return [
            _Log(level="ERROR", message="test_error", timestamp=datetime(2025, 1, 1, 10, 1, 0)),
            _Log(level="INFO", message="bot_started", timestamp=datetime(2025, 1, 1, 10, 0, 0)),
        ]

    async def get_runtime_settings(self):
        return {
            "ui_runtime_settings": {
                "investment_mode": "Balanced",
                "capital_limit_usdt": 250.0,
                "symbol_capital_limits": {"BTC/USDT": 125.0},
            }
        }

    async def close(self) -> None:
        self.closed = True


class _Settings:
    postgres_dsn = "postgresql+asyncpg://test"


def test_hydrate_state_from_database(monkeypatch) -> None:
    monkeypatch.setattr(bootstrap, "Repository", _Repo)
    state = _State()
    asyncio.run(bootstrap.hydrate_state_from_database(_Settings(), state))
    assert state.updated is not None
    assert len(state.updated["trade_history"]) == 1
    assert state.updated["trade_history"][0]["trade_id"] == 1
    assert len(state.updated["logs"]) == 2
    assert state.updated["logs"][0]["message"] == "bot_started"
    assert state.updated["runtime_settings"]["capital_limit_usdt"] == 250.0
