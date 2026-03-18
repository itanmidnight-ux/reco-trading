from __future__ import annotations

import asyncio

from reco_trading.core.bot_engine import BotEngine


def test_refresh_account_snapshot_updates_balance_equity_and_curve() -> None:
    engine = BotEngine.__new__(BotEngine)
    engine.snapshot = {
        "price": 50000.0,
        "balance": None,
        "btc_balance": 0.0,
        "btc_value": 0.0,
        "total_equity": None,
        "equity": None,
        "daily_pnl": None,
        "session_pnl": None,
        "trades_today": 0,
        "win_rate": None,
    }
    engine.trades_today = 4
    engine.win_count = 3
    engine.starting_equity = None
    engine.equity_peak = None
    engine.equity_curve_history = []

    class _Repo:
        async def get_session_pnl(self) -> float:
            return 12.5

    async def _fetch_balances() -> tuple[float, float]:
        return 1000.0, 0.01

    engine.repository = _Repo()
    engine._fetch_balances = _fetch_balances  # type: ignore[method-assign]

    asyncio.run(engine._refresh_account_snapshot(current_price=50000.0))

    assert engine.snapshot["balance"] == 1000.0
    assert engine.snapshot["btc_balance"] == 0.01
    assert engine.snapshot["btc_value"] == 500.0
    assert engine.snapshot["equity"] == 1500.0
    assert engine.snapshot["total_equity"] == 1500.0
    assert engine.snapshot["daily_pnl"] == 12.5
    assert engine.snapshot["session_pnl"] == 12.5
    assert engine.snapshot["win_rate"] == 0.75
    assert engine.equity_curve_history == [1500.0]
