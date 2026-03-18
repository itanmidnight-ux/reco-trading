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
        "confidence": 0.65,
        "balance": 1000.0,
        "equity": 1020.0,
        "btc_balance": 0.01,
        "btc_value": 500.0,
        "total_equity": 1500.0,
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
        },
        "system": {
            "exchange_status": "CONNECTED",
            "database_status": "CONNECTED",
            "bot_mode": "TESTNET",
            "api_latency_ms": 23,
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


def test_dashboard_tolerates_null_confidence_and_bad_candles() -> None:
    _app()
    tab = DashboardTab()

    snapshot = {
        "pair": "BTC/USDT",
        "status": "WAITING_MARKET_DATA",
        "trend": "NEUTRAL",
        "confidence": None,
        "daily_pnl": None,
        "win_rate": None,
        "candles_5m": [
            {"open": "bad", "high": 10, "low": 5, "close": 7, "volume": 1},
            {"open": 7, "high": 11, "low": 6, "close": 10, "volume": 2},
        ],
    }

    tab.update_state(snapshot)

    assert "BTC/USDT" in tab.top_bar.text()
    assert "Confidence 0%" in tab.confidence_label.text()
    assert "USDT" in tab.account_cards["daily_pnl"].value.text()
