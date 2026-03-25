from __future__ import annotations

from reco_trading.risk.capital_profile import CapitalProfileManager


def test_capital_profile_manager_selects_micro_profile_for_small_equity() -> None:
    manager = CapitalProfileManager()
    profile = manager.select(20.0)

    assert profile.name == "MICRO"
    assert profile.min_confidence >= 0.70


def test_capital_profile_manager_selects_large_profile_for_high_equity() -> None:
    manager = CapitalProfileManager()
    profile = manager.select(5000.0)

    assert profile.name == "LARGE"
    assert profile.max_trades_per_day >= 10


def test_operable_capital_keeps_reserve_and_buffer() -> None:
    manager = CapitalProfileManager()
    profile = manager.select(80.0)

    operable = manager.operable_capital(80.0, profile)

    assert operable < 80.0
    assert operable > 0.0
