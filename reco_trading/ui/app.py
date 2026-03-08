from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from reco_trading.ui.main_window import MainWindow
from reco_trading.ui.state_manager import StateManager


def run_gui(state_manager: StateManager) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(state_manager)
    window.show()
    return app.exec()
