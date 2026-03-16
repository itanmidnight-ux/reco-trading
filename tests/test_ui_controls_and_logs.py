from __future__ import annotations

# ruff: noqa: E402

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets", reason="PySide6 runtime deps unavailable", exc_type=ImportError)
QApplication = QtWidgets.QApplication

from reco_trading.ui.state_manager import StateManager
from reco_trading.ui.tabs.dashboard_tab import DashboardTab
from reco_trading.ui.tabs.logs_tab import LogsTab


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_dashboard_buttons_enqueue_controls() -> None:
    _app()
    manager = StateManager()
    tab = DashboardTab(state_manager=manager)

    tab.start_btn.click()
    tab.pause_btn.click()
    tab.resume_btn.click()
    tab.emergency_btn.click()
    tab.close_active_trade_btn.click()

    controls = manager.pop_control_requests()
    assert controls == ["start", "pause", "resume", "emergency_stop", "force_close"]


def test_logs_clear_button_clears_widget_and_shared_state() -> None:
    _app()
    manager = StateManager()
    manager.add_log("INFO", "first")
    manager.add_log("ERROR", "second")

    tab = LogsTab(state_manager=manager)
    tab.update_state(manager.snapshot())
    assert "second" in tab.text.toPlainText()

    tab.clear_btn.click()

    assert tab.text.toPlainText() == ""
    assert manager.snapshot().get("logs") == []

from reco_trading.ui.main_window import MainWindow


def test_main_window_applies_ui_settings_to_runtime_and_chart_visibility() -> None:
    _app()
    manager = StateManager()
    window = MainWindow(manager)

    window._on_ui_settings(
        {
            "refresh_rate_ms": 700,
            "chart_visible": False,
            "theme": "Dark+Contrast",
            "log_verbosity": "ERROR",
            "default_pair": "ETH/USDT",
        }
    )

    assert window.refresh_timer.interval() == 700
    assert window.dashboard_tab.chart_panel.isVisible() is False
    pending = manager.pop_runtime_settings()
    assert pending
    assert pending[-1]["default_pair"] == "ETH/USDT"
