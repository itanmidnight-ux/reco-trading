from __future__ import annotations

import asyncio

from reco_trading.core.bot_engine import BotEngine
from reco_trading.core.multi_pair_manager import MultiPairManager
from reco_trading.config.settings import Settings


def test_bot_engine_starts_with_btc_pair_even_if_configured_differs(monkeypatch) -> None:
    monkeypatch.setattr("reco_trading.core.bot_engine.BinanceClient", lambda *_args, **_kwargs: object())

    settings = Settings()
    settings.trading_symbol = "ETH/USDT"
    settings.trading_symbols = ["ETH/USDT", "SOL/USDT"]

    engine = BotEngine(settings)
    assert engine.symbol == "BTC/USDT"
    assert engine.symbols == ["BTC/USDT"]


def test_multi_pair_manager_does_not_scan_without_explicit_request() -> None:
    manager = MultiPairManager(exchange_client=object(), base_pairs=["BTC/USDT", "ETH/USDT"])
    calls: list[str] = []

    async def _scan() -> None:
        calls.append("scan")

    async def _select() -> None:
        calls.append("select")

    manager._scan_all_pairs = _scan  # type: ignore[method-assign]
    manager._select_best_pair = _select  # type: ignore[method-assign]

    asyncio.run(manager.start(auto_scan=False))

    assert calls == []
    assert manager._scan_task is None
    assert manager._is_running is True
