from __future__ import annotations

import asyncio
from types import SimpleNamespace

from reco_trading.core.bot_engine import BotEngine, _sanitize_runtime_settings_payload


def test_runtime_settings_sanitization_normalizes_default_pair() -> None:
    payload = _sanitize_runtime_settings_payload({"default_pair": "ethusdt"})
    assert payload["default_pair"] == "ETH/USDT"


def test_apply_runtime_settings_switches_symbol_when_safe() -> None:
    engine = BotEngine.__new__(BotEngine)
    engine.symbol = "BTC/USDT"
    engine.snapshot = {}
    engine.runtime_investment_mode = "Balanced"
    engine.runtime_dynamic_exit_enabled = False
    engine.runtime_confidence_boost_multiplier = 1.0
    engine.runtime_capital_limit_usdt = None
    engine.runtime_capital_reserve_ratio = None
    engine.runtime_min_cash_buffer_usdt = None
    engine.runtime_symbol_capital_limits = {}
    engine.runtime_risk_per_trade_fraction = None
    engine.runtime_max_trade_balance_fraction = None
    engine.settings = SimpleNamespace(capital_reserve_ratio=0.15, min_cash_buffer_usdt=10.0)
    engine._effective_risk_per_trade_fraction = lambda: 0.01  # type: ignore[method-assign]
    engine._effective_max_trade_balance_fraction = lambda: 0.20  # type: ignore[method-assign]
    engine._auto_investment_controls = lambda *_args, **_kwargs: {}  # type: ignore[method-assign]
    switched_to: list[str] = []

    async def _switch_symbol(new_symbol: str) -> None:
        switched_to.append(new_symbol)
        engine.symbol = new_symbol

    async def _has_active_trade() -> bool:
        return False

    class _Repo:
        async def set_runtime_setting(self, key: str, value: dict) -> None:
            self.last = (key, value)

    class _Logger:
        async def __call__(self, *_args, **_kwargs) -> None:
            return None

    engine._switch_symbol = _switch_symbol  # type: ignore[method-assign]
    engine._has_active_trade_for_symbol_switch = _has_active_trade  # type: ignore[method-assign]
    engine.repository = _Repo()
    engine._log = _Logger()  # type: ignore[method-assign]

    asyncio.run(
        engine._apply_runtime_settings(
            {
                "default_pair": "ETH/USDT",
                "investment_mode": "Balanced",
                "risk_per_trade_fraction": 0.01,
                "max_trade_balance_fraction": 0.2,
            }
        )
    )

    assert switched_to == ["ETH/USDT"]
    assert engine.snapshot["runtime_settings"]["default_pair"] == "ETH/USDT"


def test_apply_runtime_settings_blocks_symbol_change_when_active_trade() -> None:
    engine = BotEngine.__new__(BotEngine)
    engine.symbol = "BTC/USDT"
    engine.snapshot = {}
    engine.runtime_investment_mode = "Balanced"
    engine.runtime_dynamic_exit_enabled = False
    engine.runtime_confidence_boost_multiplier = 1.0
    engine.runtime_capital_limit_usdt = None
    engine.runtime_capital_reserve_ratio = None
    engine.runtime_min_cash_buffer_usdt = None
    engine.runtime_symbol_capital_limits = {}
    engine.runtime_risk_per_trade_fraction = None
    engine.runtime_max_trade_balance_fraction = None
    engine.settings = SimpleNamespace(capital_reserve_ratio=0.15, min_cash_buffer_usdt=10.0)
    engine._effective_risk_per_trade_fraction = lambda: 0.01  # type: ignore[method-assign]
    engine._effective_max_trade_balance_fraction = lambda: 0.20  # type: ignore[method-assign]
    engine._auto_investment_controls = lambda *_args, **_kwargs: {}  # type: ignore[method-assign]
    switched_to: list[str] = []

    async def _switch_symbol(new_symbol: str) -> None:
        switched_to.append(new_symbol)
        engine.symbol = new_symbol

    async def _has_active_trade() -> bool:
        return True

    class _Repo:
        async def set_runtime_setting(self, key: str, value: dict) -> None:
            self.last = (key, value)

    class _Logger:
        async def __call__(self, *_args, **_kwargs) -> None:
            return None

    engine._switch_symbol = _switch_symbol  # type: ignore[method-assign]
    engine._has_active_trade_for_symbol_switch = _has_active_trade  # type: ignore[method-assign]
    engine.repository = _Repo()
    engine._log = _Logger()  # type: ignore[method-assign]

    asyncio.run(engine._apply_runtime_settings({"default_pair": "ETH/USDT"}))

    assert switched_to == []
    assert engine.snapshot["runtime_settings"]["default_pair"] == "BTC/USDT"
