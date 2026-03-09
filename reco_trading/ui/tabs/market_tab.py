from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget

from reco_trading.ui.widgets.market_panel import MarketPanel


class MarketTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        self.panel = MarketPanel()
        layout.addWidget(self.panel)

    def update_state(self, state: dict) -> None:
        try:
            self.panel.update_market(state)
        except Exception:
            return
