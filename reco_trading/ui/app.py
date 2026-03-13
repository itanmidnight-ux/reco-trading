from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication
from PySide6.QtWebEngineWidgets import QWebEngineView

try:
    from PySide6.QtWebEngineQuick import QtWebEngineQuick
except Exception:  # noqa: BLE001
    QtWebEngineQuick = None

from reco_trading.ui.main_window import MainWindow
from reco_trading.ui.state_manager import StateManager


def run_gui(state_manager: StateManager) -> int:
    _ = QWebEngineView
    if QtWebEngineQuick is not None:
        QtWebEngineQuick.initialize()
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(state_manager)
    window.show()
    return app.exec()
