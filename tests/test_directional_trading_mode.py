from __future__ import annotations

import asyncio
from types import SimpleNamespace

from reco_trading.core.bot_engine import BotEngine
from reco_trading.core.state_machine import BotState
from reco_trading.exchange.binance_client import BinanceClient
from reco_trading.risk.position_manager import PositionManager


class _RiskResult:
    approved = True
    reason = "OK"


class _AdvancedRiskResult:
    approved = True
    reason = "OK"
    size_multiplier = 1.0
    pause_trading = False


class _SessionStats:
    recommendation = "NORMAL"
    current_streak = 0


def test_binance_client_uses_futures_mode_when_requested() -> None:
    client = BinanceClient(api_key="", api_secret="", testnet=True, market_type="future")
    assert client.exchange.options.get("defaultType") == "future"


def test_validate_trade_conditions_allows_sell_when_not_spot_only() -> None:
    engine = BotEngine.__new__(BotEngine)
    engine.settings = SimpleNamespace(
        spot_only_mode=False,
        confidence_threshold=0.60,
        max_drawdown_fraction=0.10,
        loss_pause_minutes=20,
    )
    engine.position_manager = PositionManager()
    engine.snapshot = {
        "balance": 1000.0,
        "total_equity": 1000.0,
        "equity": 1000.0,
        "price": 100.0,
        "session_pnl": 0.0,
    }
    engine.trades_today = 0
    engine.equity_peak = None
    engine.starting_equity = None
    engine.trading_paused_by_drawdown = False
    engine.pause_trading_until = None
    engine.consecutive_losses = 0
    engine.session_tracker = SimpleNamespace(stats=lambda: _SessionStats())
    engine.risk_manager = SimpleNamespace(validate=lambda **kwargs: _RiskResult())
    engine.advanced_risk_manager = SimpleNamespace(evaluate=lambda **kwargs: _AdvancedRiskResult())

    states: list[tuple[BotState, str | None]] = []

    async def _set_state(state: BotState, reason: str | None = None) -> None:
        states.append((state, reason))

    engine._set_state = _set_state  # type: ignore[method-assign]
    engine._is_cooldown_complete = lambda: True  # type: ignore[method-assign]

    analysis = {"side": "SELL", "confidence": 0.61}
    approved = asyncio.run(engine.validate_trade_conditions(analysis))

    assert approved is True
    assert engine.snapshot["cooldown"] == "READY"
    assert states == []
