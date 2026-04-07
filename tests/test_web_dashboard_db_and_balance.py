from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

from reco_trading.api.routes import get_balance, set_context_provider
from reco_trading.database.repository import Repository
from reco_trading.database.models import Trade
from web_site import dashboard_server


def _reset_dashboard_globals() -> None:
    dashboard_server._global_bot_instance = None
    dashboard_server._bot_instance_getter = None
    dashboard_server._fallback_repository = None


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {dashboard_server._create_jwt_token('test-user')}"}


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


def test_web_dashboard_trades_support_wins_and_losses_filters(tmp_path) -> None:
    _reset_dashboard_globals()
    db_path = tmp_path / "dashboard_filters.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    os.environ["DASHBOARD_AUTH_ENABLED"] = "false"

    repo = Repository(f"sqlite+aiosqlite:///{db_path}")
    asyncio.run(repo.setup())
    winner = asyncio.run(
        repo.create_trade(
            symbol="BTC/USDT",
            side="BUY",
            quantity=1.0,
            entry_price=100.0,
            stop_loss=90.0,
            take_profit=120.0,
            order_id="winner",
        )
    )
    loser = asyncio.run(
        repo.create_trade(
            symbol="BTC/USDT",
            side="BUY",
            quantity=1.0,
            entry_price=100.0,
            stop_loss=90.0,
            take_profit=120.0,
            order_id="loser",
        )
    )
    async def _close_trades() -> None:
        async with repo.session_factory() as session:
            for trade_id, exit_price, pnl in ((winner.id, 110.0, 10.0), (loser.id, 95.0, -5.0)):
                trade = await session.get(Trade, trade_id)
                trade.exit_price = exit_price
                trade.pnl = pnl
                trade.status = "CLOSED"
                trade.close_timestamp = trade.timestamp
            await session.commit()

    asyncio.run(_close_trades())
    asyncio.run(repo.close())

    app = dashboard_server.create_app()
    client = app.test_client()

    wins_response = client.get("/api/trades?pnl_positive=true", headers=_auth_headers())
    wins_payload = wins_response.get_json()
    assert wins_response.status_code == 200
    assert len(wins_payload["trades"]) == 1
    assert wins_payload["trades"][0]["pnl"] > 0

    losses_response = client.get("/api/trades?pnl_negative=true", headers=_auth_headers())
    losses_payload = losses_response.get_json()
    assert losses_response.status_code == 200
    assert len(losses_payload["trades"]) == 1
    assert losses_payload["trades"][0]["pnl"] < 0

    os.environ.pop("DATABASE_URL", None)
    os.environ.pop("DASHBOARD_AUTH_ENABLED", None)
    _reset_dashboard_globals()


def test_web_dashboard_settings_are_persisted(tmp_path) -> None:
    _reset_dashboard_globals()
    db_path = tmp_path / "dashboard_settings.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    os.environ["DASHBOARD_AUTH_ENABLED"] = "false"

    repo = Repository(f"sqlite+aiosqlite:///{db_path}")
    asyncio.run(repo.setup())

    class DummySettings:
        risk_per_trade_fraction = 0.01
        daily_loss_limit_fraction = 0.03
        max_drawdown_fraction = 0.10
        max_trades_per_day = 10
        min_signal_confidence = 0.60
        adx_min_threshold = 20.0
        primary_timeframe = "5m"
        loop_sleep_seconds = 1
        max_trade_balance_fraction = 0.2
        spot_only_mode = True

    bot = SimpleNamespace(
        repository=repo,
        settings=DummySettings(),
        symbol="BTC/USDT",
        snapshot={},
    )
    dashboard_server.set_bot_instance(bot)
    app = dashboard_server.create_app()
    client = app.test_client()

    settings_response = client.post(
        "/api/settings",
        headers=_auth_headers(),
        json={"risk_per_trade": 0.02, "primary_timeframe": "15m", "loop_sleep_seconds": 3},
    )
    assert settings_response.status_code == 200
    assert bot.settings.risk_per_trade_fraction == 0.02
    assert bot.settings.primary_timeframe == "15m"
    assert bot.settings.loop_sleep_seconds == 3

    pair_response = client.post("/api/settings/pair", headers=_auth_headers(), json={"symbol": "ETH/USDT"})
    assert pair_response.status_code == 200

    runtime_settings = asyncio.run(repo.get_runtime_settings())
    dashboard_settings = runtime_settings.get("web_dashboard_settings", {})
    assert dashboard_settings["symbol"] == "ETH/USDT"
    assert dashboard_settings["risk_per_trade"] == 0.02
    assert dashboard_settings["primary_timeframe"] == "15m"

    asyncio.run(repo.close())
    os.environ.pop("DATABASE_URL", None)
    os.environ.pop("DASHBOARD_AUTH_ENABLED", None)
    _reset_dashboard_globals()
