from __future__ import annotations

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
from reco_trading.ui.tabs.settings_tab import SettingsTab
from reco_trading.ui.tabs.trades_tab import TradesTab


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


def test_risk_tab_drawdown_alert_uses_pipeline_daily_drawdown() -> None:
    _app()
    tab = RiskTab()

    tab.update_state({"risk_metrics": {"current_exposure": 0.2, "daily_drawdown": "0.0400"}})
    alerts = [tab.alerts.item(i).text() for i in range(tab.alerts.count())]
    assert any("Drawdown exceeded 3%" in msg for msg in alerts)


def test_system_tab_preserves_default_version_when_state_lacks_bot_version() -> None:
    _app()
    tab = SystemTab()

    tab.update_state({"system": {"exchange_status": "OK", "database_status": "OK"}})
    assert tab.cards["version"].value.text() == "reco-trading"


def test_settings_tab_emits_api_keys_and_runtime_settings() -> None:
    _app()
    tab = SettingsTab()
    captured: list[dict] = []
    tab.settings_changed.connect(captured.append)

    tab.refresh_rate.setValue(1500)
    tab.api_key.setText("demo_key")
    tab.api_secret.setText("demo_secret")
    tab._emit()

    assert captured, "settings_changed should emit payload"
    payload = captured[-1]
    assert payload["refresh_rate_ms"] == 1500
    assert payload["binance_api_key"] == "demo_key"
    assert payload["binance_api_secret"] == "demo_secret"


def test_trades_tab_kpis_update_from_trade_history() -> None:
    _app()
    tab = TradesTab()
    tab.update_state(
        {
            "trade_history": [
                {"trade_id": 1, "status": "OPEN", "pnl": None, "pair": "BTC/USDT", "side": "BUY"},
                {"trade_id": 2, "status": "TAKE_PROFIT_HIT", "pnl": 12.5, "pair": "BTC/USDT", "side": "BUY"},
                {"trade_id": 3, "status": "STOP_LOSS_HIT", "pnl": -5.0, "pair": "BTC/USDT", "side": "SELL"},
            ]
        }
    )

    assert tab.summary_cards["total"].value.text() == "3"
    assert tab.summary_cards["open"].value.text() == "1"
    assert tab.summary_cards["closed"].value.text() == "2"
    assert tab.summary_cards["realized_pnl"].value.text() == "7.5000"
