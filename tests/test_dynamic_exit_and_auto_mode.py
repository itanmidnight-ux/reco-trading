from __future__ import annotations

from types import SimpleNamespace

from reco_trading.core.bot_engine import BotEngine, _sanitize_runtime_settings_payload
from reco_trading.risk.capital_profile import CapitalProfileManager
from reco_trading.risk.investment_optimizer import InvestmentOptimizer
from reco_trading.risk.position_manager import Position, PositionManager


def test_runtime_settings_sanitization_includes_dynamic_exit_flag() -> None:
    payload = _sanitize_runtime_settings_payload({"investment_mode": "Auto-Optimized", "dynamic_exit_enabled": True})
    assert payload["investment_mode"] == "Auto-Optimized"
    assert payload["dynamic_exit_enabled"] is True


def test_auto_investment_controls_generates_capital_aware_values() -> None:
    engine = BotEngine.__new__(BotEngine)
    engine.snapshot = {"total_equity": 240.0, "equity": 240.0, "balance": 240.0}
    engine.investment_optimizer = InvestmentOptimizer()
    engine.capital_profile_manager = CapitalProfileManager()
    engine.settings = SimpleNamespace(enable_capital_profiles=True)
    controls = BotEngine._auto_investment_controls(engine, "Auto-Optimized")

    assert controls["dynamic_exit_enabled"] is True
    assert 0.001 <= float(controls["risk_per_trade_fraction"]) <= 0.10
    assert 0.01 <= float(controls["max_trade_balance_fraction"]) <= 1.0
    assert float(controls["capital_limit_usdt"]) >= 0.0
    assert "optimization_reason" in controls


def test_investment_optimizer_respects_user_caps() -> None:
    profile = CapitalProfileManager().select(400.0)
    controls = InvestmentOptimizer().optimize(
        equity=400.0,
        profile=profile,
        volatility_ratio=0.01,
        drawdown_fraction=0.05,
        win_rate=0.6,
        risk_cap=0.003,
        allocation_cap=0.08,
    )
    assert controls.risk_per_trade_fraction <= 0.003
    assert controls.max_trade_balance_fraction <= 0.08
    assert controls.dynamic_exit_enabled is True


def test_investment_optimizer_handles_extreme_capital_ranges() -> None:
    optimizer = InvestmentOptimizer()
    low_profile = CapitalProfileManager().select(5.0)
    high_profile = CapitalProfileManager().select(500_000.0)
    low_controls = optimizer.optimize(
        equity=5.0,
        profile=low_profile,
        volatility_ratio=0.03,
        drawdown_fraction=0.10,
        win_rate=0.5,
    )
    high_controls = optimizer.optimize(
        equity=500_000.0,
        profile=high_profile,
        volatility_ratio=0.01,
        drawdown_fraction=0.02,
        win_rate=0.6,
    )
    assert low_controls.capital_limit_usdt >= 0.0
    assert low_controls.risk_per_trade_fraction <= 0.0035
    assert high_controls.risk_per_trade_fraction <= 0.0030
    assert high_controls.max_trade_balance_fraction <= 0.07


def test_position_manager_dynamic_exit_closes_on_peak_retrace() -> None:
    manager = PositionManager()
    position = Position(
        trade_id=1,
        side="BUY",
        quantity=0.1,
        entry_price=100.0,
        stop_loss=98.0,
        take_profit=120.0,
        atr=1.0,
        dynamic_exit_enabled=True,
        peak_retrace_atr=0.8,
    )
    manager.open(position)
    assert manager.check_exit(position, 103.0) is None
    assert manager.check_exit(position, 105.0) is None
    assert manager.check_exit(position, 104.1) in {"PEAK_RETRACE_EXIT", "TRAILING_STOP_HIT"}


def test_position_manager_sets_entry_timestamp_when_missing() -> None:
    manager = PositionManager()
    position = Position(
        trade_id=2,
        side="BUY",
        quantity=0.1,
        entry_price=100.0,
        stop_loss=98.0,
        take_profit=120.0,
        atr=1.0,
    )
    manager.open(position)
    assert position.entry_timestamp_ms is not None
    assert position.entry_timestamp_ms > 0


def test_per_trade_investment_controls_adjust_to_confidence_and_volatility() -> None:
    engine = BotEngine.__new__(BotEngine)
    engine.runtime_investment_mode = "Auto-Optimized"
    engine.snapshot = {
        "optimized_risk_per_trade_fraction": 0.01,
        "optimized_max_trade_balance_fraction": 0.20,
        "total_equity": 500.0,
        "equity": 500.0,
    }
    engine.equity_peak = 520.0
    engine.runtime_risk_per_trade_fraction = 0.01
    engine.runtime_max_trade_balance_fraction = 0.20
    engine.capital_profile_manager = CapitalProfileManager()
    engine.settings = SimpleNamespace(
        enable_capital_profiles=True,
        risk_per_trade_fraction=0.01,
        max_trade_balance_fraction=0.20,
    )
    conservative = BotEngine._compute_per_trade_investment_controls(engine, confidence=0.62, price=100.0, atr=2.5)
    aggressive = BotEngine._compute_per_trade_investment_controls(engine, confidence=0.92, price=100.0, atr=0.4)
    assert conservative["risk_per_trade_fraction"] < aggressive["risk_per_trade_fraction"]
    assert conservative["max_trade_balance_fraction"] < aggressive["max_trade_balance_fraction"]
