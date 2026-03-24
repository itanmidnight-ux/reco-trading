from __future__ import annotations

# ruff: noqa: E402

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PySide6.QtWidgets", reason="PySide6 runtime deps unavailable", exc_type=ImportError)
QApplication = QtWidgets.QApplication

from reco_trading.ui.tabs.analytics_tab import AnalyticsTab
from reco_trading.ui.tabs.dashboard_tab import DashboardTab
from reco_trading.ui.tabs.market_tab import MarketTab
from reco_trading.ui.tabs.risk_tab import RiskTab
from reco_trading.ui.tabs.system_tab import SystemTab


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_tabs_accept_full_snapshot_without_errors() -> None:
    _app()
    snapshot = {
        "pair": "BTC/USDT",
        "price": 50000.0,
        "current_price": 50000.0,
        "trend": "BUY",
        "status": "WAITING_MARKET_DATA",
        "spread": 2.0,
        "adx": 23.5,
        "volatility_regime": "NORMAL_VOLATILITY",
        "order_flow": "BUY",
        "signal": "HOLD",
        "raw_signal": "BUY",
        "confidence": 0.65,
        "signal_quality_score": 0.82,
        "balance": 1000.0,
        "equity": 1020.0,
        "btc_balance": 0.01,
        "btc_value": 500.0,
        "total_equity": 1500.0,
        "capital_profile": "MEDIUM",
        "operable_capital_usdt": 1200.0,
        "capital_reserve_ratio": 0.15,
        "min_cash_buffer_usdt": 10.0,
        "daily_pnl": -12.5,
        "position_side": "BUY",
        "entry_price": 49800.0,
        "position_size": 0.02,
        "unrealized_pnl": 4.2,
        "trades_today": 3,
        "win_rate": 0.66,
        "logs": [{"time": "12:00:00", "message": "ok"}],
        "bid": 49999.5,
        "ask": 50000.5,
        "volume": 150.0,
        "atr": 120.0,
        "risk_metrics": {
            "risk_per_trade": "1.00%",
            "max_concurrent_trades": 1,
            "daily_drawdown": -0.02,
            "current_exposure": 0.35,
            "capital_profile": "MEDIUM",
            "operable_capital_usdt": 1200.0,
            "setup_quality_score": 0.82,
            "adaptive_size_multiplier": 0.95,
        },
        "system": {
            "exchange_status": "CONNECTED",
            "database_status": "CONNECTED",
            "bot_mode": "TESTNET",
            "api_latency_ms": 23,
            "ui_render_ms": 18,
            "ui_staleness_ms": 40,
            "ui_lag_detected": False,
            "memory_usage_mb": 120,
            "redis_status": "UNKNOWN",
            "uptime_seconds": 100,
            "last_server_sync": "2026-01-01T00:00:00",
        },
        "analytics": {"equity_curve": [1000.0, 1010.0], "win_rate": 0.66, "total_trades": 3},
    }

    DashboardTab().update_state(snapshot)
    MarketTab().update_state(snapshot)
    RiskTab().update_state(snapshot)
    SystemTab().update_state(snapshot)
    AnalyticsTab().update_state(snapshot)

def test_dashboard_controls_and_status_colors_match_engine_states() -> None:
    _app()
    tab = DashboardTab()

    tab.update_state({"status": "paused"})
    assert tab.resume_btn.isVisible()
    assert not tab.pause_btn.isVisible()

    tab.update_state({"status": "position_open"})
    assert tab.pause_btn.isVisible()
    assert not tab.start_btn.isVisible()

    tab.update_state({"status": "waiting_market_data"})
    assert "#f0b90b" in tab.top_bar.styleSheet()

    tab.update_state({"status": "error"})
    assert "#ea3943" in tab.top_bar.styleSheet()


def test_dashboard_highlight_cards_and_feed_render_rich_snapshot() -> None:
    _app()
    tab = DashboardTab()
    tab.update_state(
        {
            "pair": "BTC/USDT",
            "current_price": 50123.45,
            "trend": "BUY",
            "signal": "BUY",
            "confidence": 0.87,
            "daily_pnl": 42.5,
            "cooldown": "READY",
            "status": "waiting_market_data",
            "risk_metrics": {"current_exposure": 0.35},
            "system": {"api_latency_ms": 28, "ui_render_ms": 14, "ui_staleness_ms": 32, "ui_lag_detected": False},
            "runtime_settings": {"investment_mode": "Balanced", "capital_limit_usdt": 300.0},
            "logs": [{"time": "12:00:00", "level": "INFO", "message": "signal generated"}],
            "capital_profile": "SMALL",
            "operable_capital_usdt": 180.0,
            "capital_reserve_ratio": 0.25,
            "min_cash_buffer_usdt": 7.5,
            "raw_signal": "BUY",
            "signal_quality_score": 0.91,
            "risk_metrics": {
                "current_exposure": 0.35,
                "adaptive_size_multiplier": 0.85,
                "advanced_size_multiplier": 0.90,
                "advanced_risk_reason": "OK",
            },
        }
    )

    assert "BUY • 87%" in tab.hero_cards["signal"].value.text()
    assert "UI 14" in tab.feed_meta.text()
    assert "Balanced" in tab.feed_meta.text()
    assert "Profile SMALL" in tab.feed_meta.text()
    assert "SMALL" in tab.hero_cards["capital"].value.text()
    assert "180.00 USDT" in tab.hero_cards["operable"].value.text()
    assert "Quality 91%" in tab.execution_insight.text()
    assert "signal generated" in tab.feed.text()


def test_analytics_tab_derives_kpis_from_trade_history_when_payload_is_partial() -> None:
    _app()
    tab = AnalyticsTab()
    tab.update_state(
        {
            "equity": 1200.0,
            "confidence": 0.8,
            "trade_history": [
                {"trade_id": 1, "status": "TAKE_PROFIT_HIT", "pnl": 25.0, "entry_slippage_ratio": 0.001},
                {"trade_id": 2, "status": "STOP_LOSS_HIT", "pnl": -10.0, "exit_slippage_ratio": 0.002},
            ],
            "analytics": {"equity_curve": [1000.0, 1025.0, 1015.0]},
        }
    )

    assert tab.cards["total_trades"].value.text() == "2"
    assert tab.cards["win_rate"].value.text() == "50.00%"
    assert tab.cards["profit_factor"].value.text() == "2.5"
