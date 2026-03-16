from __future__ import annotations

# ruff: noqa: E402

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtCore", reason="PySide6 runtime deps unavailable", exc_type=ImportError)
QtWidgets = pytest.importorskip("PySide6.QtWidgets", reason="PySide6 runtime deps unavailable", exc_type=ImportError)
QApplication = QtWidgets.QApplication

from reco_trading.ui.state_manager import StateManager
from reco_trading.ui.tabs.settings_tab import SettingsTab


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_state_manager_runtime_settings_queue() -> None:
    _app()
    manager = StateManager()
    manager.push_runtime_settings({"risk_per_trade_fraction": 0.01})
    manager.push_runtime_settings({"max_trade_balance_fraction": 0.2})
    updates = manager.pop_runtime_settings()
    assert len(updates) == 2
    assert updates[0]["risk_per_trade_fraction"] == 0.01
    assert manager.pop_runtime_settings() == []


def test_settings_tab_emits_investment_payload() -> None:
    _app()
    tab = SettingsTab()
    captured: list[dict] = []
    tab.settings_changed.connect(captured.append)

    tab.investment_mode.setCurrentText("Custom")
    tab.capital_limit.setValue(250.0)
    tab.risk_per_trade.setValue(1.5)
    tab.max_allocation.setValue(30.0)
    tab._emit()

    assert captured
    payload = captured[-1]
    assert payload["investment_mode"] == "Custom"
    assert payload["capital_limit_usdt"] == 250.0
    assert payload["risk_per_trade_fraction"] == pytest.approx(0.015)
    assert payload["max_trade_balance_fraction"] == pytest.approx(0.30)


def test_settings_tab_capital_limit_and_pair_budget_are_distinct() -> None:
    _app()
    tab = SettingsTab()

    tab.capital_limit.setValue(500.0)
    tab.symbol_budget.setValue(120.0)
    tab.max_allocation.setValue(25.0)
    tab._emit()

    text = tab.capital_breakdown.text()
    assert "Global capital limit: 500.00" in text
    assert "Per-pair budget: 120.00" in text
    assert "Effective safety cap: 120.00" in text
    assert "30.00 USDT" in tab.simulation_hint.text()


def test_settings_tab_initialization_has_no_name_errors() -> None:
    _app()
    tab = SettingsTab()
    assert tab.refresh_rate.value() >= 250
