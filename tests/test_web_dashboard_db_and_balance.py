from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

from reco_trading.api.routes import get_balance, set_context_provider
from reco_trading.database.repository import Repository
from web_site import dashboard_server


def _reset_dashboard_globals() -> None:
    dashboard_server._global_bot_instance = None
    dashboard_server._bot_instance_getter = None
    dashboard_server._fallback_repository = None


def test_web_dashboard_reads_trades_from_database_without_bot(tmp_path) -> None:
    _reset_dashboard_globals()
    db_path = tmp_path / "dashboard_test.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    os.environ["DASHBOARD_AUTH_ENABLED"] = "false"

    repo = Repository(f"sqlite+aiosqlite:///{db_path}")
    asyncio.run(repo.setup())
    asyncio.run(
        repo.create_trade(
            symbol="BTC/USDT",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit=52000.0,
            order_id="test-order",
        )
    )
    asyncio.run(repo.close())

    app = dashboard_server.create_app()
    client = app.test_client()
    response = client.get("/api/all_trades")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["source"] == "database"
    assert payload["total"] >= 1
    assert payload["trades"][0]["pair"] == "BTC/USDT"

    os.environ.pop("DATABASE_URL", None)
    os.environ.pop("DASHBOARD_AUTH_ENABLED", None)
    _reset_dashboard_globals()


def test_api_balance_uses_runtime_snapshot_values() -> None:
    set_context_provider(
        lambda: SimpleNamespace(
            snapshot={
                "balance": 123.45,
                "equity": 200.0,
                "total_equity": 250.0,
            }
        )
    )

    payload = asyncio.run(get_balance())
    assert payload["total"] == 250.0
    assert payload["free"] == 123.45
    assert payload["locked"] == 126.55


def test_web_dashboard_stop_trade_control_queues_force_close() -> None:
    _reset_dashboard_globals()
    os.environ["DASHBOARD_AUTH_ENABLED"] = "false"

    class DummyStateManager:
        def __init__(self) -> None:
            self.called = False

        def request_force_close(self) -> None:
            self.called = True

    bot = SimpleNamespace(
        state_manager=DummyStateManager(),
        snapshot={},
    )
    dashboard_server.set_bot_instance(bot)
    app = dashboard_server.create_app()
    client = app.test_client()

    response = client.post("/api/control/stop_trade")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["success"] is True
    assert bot.state_manager.called is True
    
    os.environ.pop("DASHBOARD_AUTH_ENABLED", None)
    _reset_dashboard_globals()
