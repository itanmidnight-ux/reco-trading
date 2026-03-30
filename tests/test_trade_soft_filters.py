from __future__ import annotations

from reco_trading.core.bot_engine import BotEngine
from reco_trading.risk.capital_profile import CapitalProfile


class _Settings:
    enforce_fee_floor = True
    estimated_fee_rate = 0.001
    min_expected_reward_risk = 1.8


def _profile() -> CapitalProfile:
    return CapitalProfile(
        name="MEDIUM",
        min_equity=100.0,
        max_equity=1000.0,
        reserve_ratio=0.15,
        reserve_buffer_usdt=15.0,
        risk_per_trade_fraction=0.0075,
        max_trade_balance_fraction=0.20,
        min_confidence=0.60,
        max_trades_per_day=6,
        cooldown_minutes=5,
        loss_pause_minutes=25,
        loss_pause_after_consecutive=3,
        max_spread_ratio=0.0030,
        min_expected_reward_risk=2.0,
        min_operable_notional_buffer=1.10,
        max_concurrent_trades=1,
        entry_quality_floor=0.65,
        size_multiplier=0.90,
    )


def _engine() -> BotEngine:
    engine = BotEngine.__new__(BotEngine)
    engine.settings = _Settings()
    engine.snapshot = {"price": 100.0, "spread": 0.35, "atr": 0.90}
    engine._effective_max_spread_ratio = lambda: 0.003  # type: ignore[attr-defined]
    engine._current_capital_profile = _profile  # type: ignore[attr-defined]
    return engine


def test_spread_gate_allows_soft_reduction_before_hard_reject() -> None:
    engine = _engine()

    soft = engine._assess_spread_gate(spread=0.34, price=100.0)
    hard = engine._assess_spread_gate(spread=0.45, price=100.0)

    assert soft["approved"] is True
    assert float(soft["size_multiplier"]) < 1.0
    assert hard["approved"] is False


def test_trade_cost_gate_softens_marginal_setups() -> None:
    engine = _engine()

    soft = engine._assess_trade_cost_gate("BUY")
    engine.snapshot["atr"] = 0.60
    hard = engine._assess_trade_cost_gate("BUY")

    assert soft["approved"] is True
    assert float(soft["size_multiplier"]) < 1.0
    assert hard["approved"] is False
