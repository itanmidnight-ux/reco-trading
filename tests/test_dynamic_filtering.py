from __future__ import annotations

from types import SimpleNamespace

from reco_trading.core.bot_engine import BotEngine
from reco_trading.risk.capital_profile import CapitalProfileManager


def _engine() -> BotEngine:
    engine = BotEngine.__new__(BotEngine)
    engine.settings = SimpleNamespace(
        max_trades_per_day=6,
        max_spread_ratio=0.003,
        min_signal_confidence=0.62,
        confidence_threshold=0.62,
        enable_capital_profiles=True,
    )
    engine.snapshot = {
        "adx": 28.0,
        "price": 100.0,
        "spread": 0.10,
        "signal_quality_score": 0.9,
        "confluence_aligned": True,
    }
    engine.capital_profile_manager = CapitalProfileManager()
    engine.runtime_capital_limit_usdt = None
    engine.runtime_symbol_capital_limits = {}
    engine.symbol = "BTCUSDT"
    return engine


def test_opportunity_score_increases_for_high_quality_setups() -> None:
    engine = _engine()
    high = engine._opportunity_score({"setup_quality": 0.92})

    engine.snapshot["adx"] = 16.0
    engine.snapshot["spread"] = 0.5
    engine.snapshot["confluence_aligned"] = False
    low = engine._opportunity_score({"setup_quality": 0.70})

    assert high > low
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0


def test_dynamic_threshold_and_trade_cap_adapt_with_opportunity() -> None:
    engine = _engine()
    base = engine._effective_max_trades_per_day()
    high_score = 0.95
    low_score = 0.10

    assert engine._dynamic_min_confidence_threshold(high_score) <= engine._dynamic_min_confidence_threshold(low_score)
    assert engine._dynamic_max_trades_per_day(high_score) >= base


def test_capital_profile_policy_changes_strictness_and_throughput() -> None:
    engine = _engine()
    engine.snapshot["total_equity_usdt"] = 10.0  # MICRO profile
    micro_conf = engine._dynamic_min_confidence_threshold(0.5)
    micro_trades = engine._dynamic_max_trades_per_day(1.0)

    engine.snapshot["total_equity_usdt"] = 5_000.0  # LARGE profile
    large_conf = engine._dynamic_min_confidence_threshold(0.5)
    large_trades = engine._dynamic_max_trades_per_day(1.0)

    assert micro_conf > large_conf
    assert micro_trades <= large_trades
