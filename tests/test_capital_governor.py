from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reco_trading.core.institutional_risk_manager import InstitutionalRiskManager, RiskLimits
from reco_trading.core.portfolio_optimization import ConvexPortfolioOptimizer
from reco_trading.execution.smart_order_router import SmartOrderRouter, VenueSnapshot
from reco_trading.kernel.capital_governor import CapitalGovernor
from reco_trading.monitoring.alert_manager import AlertManager
from reco_trading.monitoring.metrics import TradingMetrics


class _StubMetrics(TradingMetrics):
    def __init__(self) -> None:
        super().__init__()
        self.requests = 0
        self.errors = 0

    def observe_request(self, component: str, **labels: str) -> None:
        self.requests += 1

    def observe_error(self, component: str, error_type: str, **labels: str) -> None:
        self.errors += 1


class _StubAlerts(AlertManager):
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def emit(self, title: str, detail: str, *, severity: str = 'error', exchange: str | None = None, payload=None) -> None:
        self.events.append((title, detail))


def test_rolling_var_cvar_returns_positive_losses() -> None:
    governor = CapitalGovernor(hard_cap_global=1000)
    returns = np.array([0.01, -0.02, 0.005, -0.03, -0.01, 0.015])
    var, cvar = governor.rolling_var_cvar(returns, alpha=0.8)
    assert var >= 0
    assert cvar >= var


def test_governor_rejection_records_metrics_and_alert() -> None:
    metrics = _StubMetrics()
    alerts = _StubAlerts()
    governor = CapitalGovernor(hard_cap_global=0.0, metrics=metrics, alert_manager=alerts)
    governor.update_state(
        strategy='mm',
        exchange='binance',
        symbol='BTCUSDT',
        capital_by_strategy=100,
        capital_by_exchange=100,
        total_exposure=100,
        asset_exposure=100,
    )
    ticket = governor.issue_ticket(
        strategy='mm',
        exchange='binance',
        symbol='BTCUSDT',
        requested_notional=10,
        pnl_or_returns=[0.0, -0.01],
        spread_bps=5,
        available_liquidity=100,
        price_gap_pct=0.01,
    )
    assert ticket.status == 'rejected'
    assert metrics.errors == 1
    assert alerts.events


def test_risk_portfolio_router_require_valid_ticket_when_governor_enabled() -> None:
    governor = CapitalGovernor(hard_cap_global=10_000)
    governor.update_state(
        strategy='arb',
        exchange='binance',
        symbol='ETH/USDT',
        capital_by_strategy=1000,
        capital_by_exchange=1000,
        total_exposure=100,
        asset_exposure=100,
    )
    ticket = governor.issue_ticket(
        strategy='arb',
        exchange='binance',
        symbol='ETH/USDT',
        requested_notional=50,
        pnl_or_returns=[0.001, -0.001, 0.002],
        spread_bps=10,
        available_liquidity=1000,
        price_gap_pct=0.005,
    )

    risk = InstitutionalRiskManager(
        RiskLimits(
            risk_per_trade=0.01,
            max_daily_loss=0.03,
            max_global_drawdown=0.2,
            max_total_exposure=0.6,
            max_asset_exposure=0.3,
            correlation_threshold=0.8,
        ),
        capital_governor=governor,
    )
    assessment = risk.assess(
        symbol='ETH/USDT',
        side='BUY',
        equity=1000,
        daily_pnl=10,
        current_price=100,
        atr=2,
        annualized_volatility=0.3,
        volatility_multiplier=1.0,
        expected_win_rate=0.6,
        avg_win=2.0,
        avg_loss=1.0,
        returns_matrix=pd.DataFrame({'ETH/USDT': [0.001, -0.001, 0.002]}),
        exchange='binance',
        notional_by_exchange={'binance': 50},
        total_exposure=100,
        capital_ticket=ticket,
    )
    assert assessment.allowed is True

    optimizer = ConvexPortfolioOptimizer(capital_governor=governor)
    out = optimizer.mean_variance(
        pd.DataFrame({'A': [0.01, 0.02, -0.01], 'B': [0.008, 0.01, -0.005]}),
        target_return=0.001,
        capital_ticket=ticket,
    )
    assert abs(sum(out.weights.values()) - 1.0) < 1e-5

    router = SmartOrderRouter(capital_governor=governor)
    routed = router.route_order(
        amount=5,
        venues=[
            VenueSnapshot('v1', spread_bps=10, depth=100, latency_ms=10, fee_bps=5, fill_ratio=0.95, liquidity=100),
            VenueSnapshot('v2', spread_bps=11, depth=90, latency_ms=12, fee_bps=6, fill_ratio=0.94, liquidity=80),
        ],
        capital_ticket=ticket,
    )
    assert routed

    with pytest.raises(ValueError, match='capital_ticket_invalid'):
        router.route_order(
            amount=5,
            venues=[VenueSnapshot('v1', spread_bps=10, depth=100, latency_ms=10, fee_bps=5, fill_ratio=0.95, liquidity=100)],
            capital_ticket=None,
        )
