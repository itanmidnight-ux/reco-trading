from __future__ import annotations

# ruff: noqa: E402

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets", reason="PySide6 runtime deps unavailable", exc_type=ImportError)
QApplication = QtWidgets.QApplication

from reco_trading.ui.main_window import MainWindow
from reco_trading.ui.state_manager import StateManager


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_main_window_initializes_without_tab_elide_type_error() -> None:
    _app()
    manager = StateManager()
    window = MainWindow(manager)
    assert window.tabs.count() == 9
