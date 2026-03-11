from __future__ import annotations

from reco_trading.risk.advanced_risk_manager import AdvancedRiskManager


def test_advanced_risk_manager_daily_loss_blocks() -> None:
    mgr = AdvancedRiskManager(max_daily_loss_percent=2.0)
    decision = mgr.evaluate(
        daily_pnl=-30.0,
        starting_equity=1000.0,
        consecutive_losses=0,
        current_equity=970.0,
        peak_equity=1000.0,
        volatility_ratio=0.01,
    )
    assert decision.approved is False
    assert decision.pause_trading is True


def test_dynamic_position_sizing_applies_multiplier() -> None:
    assert AdvancedRiskManager.dynamic_position_size(1.2, 0.5) == 0.6
