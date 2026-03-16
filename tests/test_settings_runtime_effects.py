from __future__ import annotations

from types import SimpleNamespace

from reco_trading.core.bot_engine import BotEngine


def test_apply_runtime_settings_updates_symbol_timeframe_and_caps() -> None:
    engine = BotEngine.__new__(BotEngine)
    engine.runtime_investment_mode = "Balanced"
    engine.runtime_risk_per_trade_fraction = None
    engine.runtime_max_trade_balance_fraction = None
    engine.runtime_capital_limit_usdt = None
    engine.symbol = "BTC/USDT"
    engine.order_manager = SimpleNamespace(symbol="BTC/USDT")
    engine.market_stream = SimpleNamespace(symbol="BTC/USDT")
    engine.snapshot = {}
    engine._cached_frame5 = object()
    engine._cached_frame15 = object()
    engine._last_primary_indicator_ts = object()
    engine._last_confirmation_indicator_ts = object()
    engine.settings = SimpleNamespace(timeframe="5m", confirmation_timeframe="15m", risk_per_trade_fraction=0.01, max_trade_balance_fraction=0.2)

    payload = {
        "investment_mode": "Aggressive",
        "risk_per_trade_fraction": 0.03,
        "max_trade_balance_fraction": 0.4,
        "capital_limit_usdt": 300.0,
        "default_pair": "ETH/USDT",
        "default_timeframe": "1m / 5m",
        "symbol_capital_limits": {"ETH/USDT": 125.0},
        "theme": "Dark+Contrast",
        "log_verbosity": "DEBUG",
        "chart_visible": False,
        "refresh_rate_ms": 750,
    }

    BotEngine._apply_runtime_settings(engine, payload)

    assert engine.runtime_investment_mode == "Aggressive"
    assert engine.runtime_risk_per_trade_fraction == 0.03
    assert engine.runtime_max_trade_balance_fraction == 0.4
    assert engine.runtime_capital_limit_usdt == 125.0
    assert engine.symbol == "ETH/USDT"
    assert engine.order_manager.symbol == "ETH/USDT"
    assert engine.market_stream.symbol == "ETH/USDT"
    assert engine.settings.timeframe == "1m"
    assert engine.settings.confirmation_timeframe == "5m"
    assert engine.snapshot["timeframe"] == "1m / 5m"
    assert engine.snapshot["runtime_settings"]["theme"] == "Dark+Contrast"
    assert engine.snapshot["runtime_settings"]["refresh_rate_ms"] == 750
