from __future__ import annotations

from reco_trading.config.settings import Settings
from reco_trading.risk.portfolio_risk import PortfolioRiskController


def test_portfolio_risk_controller_enforces_caps_and_correlation() -> None:
    ctrl = PortfolioRiskController(
        max_global_exposure_fraction=0.5,
        max_symbol_correlation=0.8,
        symbol_caps={"BTCUSDT": 200.0},
    )
    denied_cap = ctrl.validate(
        symbol="BTCUSDT",
        requested_notional=80.0,
        current_symbol_notional=150.0,
        total_open_notional=150.0,
        equity=1000.0,
        max_correlation_observed=0.4,
    )
    assert not denied_cap.approved
    assert denied_cap.reason == "symbol_cap_limit"

    denied_corr = ctrl.validate(
        symbol="ETHUSDT",
        requested_notional=100.0,
        current_symbol_notional=0.0,
        total_open_notional=200.0,
        equity=1000.0,
        max_correlation_observed=0.95,
    )
    assert not denied_corr.approved
    assert denied_corr.reason == "correlation_limit"


def test_settings_normalizes_symbol_caps() -> None:
    settings = Settings(
        postgres_dsn="postgresql+asyncpg://u:p@localhost/db",
        SYMBOL_CAPITAL_LIMITS={"eth/usdt": "120", "bad": -1},
    )
    assert settings.symbol_capital_limits == {"ETHUSDT": 120.0}
