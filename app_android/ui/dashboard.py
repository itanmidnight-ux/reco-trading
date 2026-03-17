from __future__ import annotations

from kivy.clock import Clock
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label

from config import REFRESH_INTERVAL_SECONDS
from services.api_client import APIClient
from ui.components import ActionButton, Card


class Dashboard(BoxLayout):
    def __init__(self, **kwargs) -> None:
        super().__init__(orientation="vertical", padding=dp(16), spacing=dp(12), **kwargs)
        self.client = APIClient()

        self.header = Label(text="Reco Trading Control", font_size="22sp", bold=True, size_hint_y=None, height=dp(42))
        self.status = Card("Bot Status")
        self.balance = Card("Balance")
        self.pnl = Card("Daily PnL")
        self.positions = Card("Trades Activos")

        self.pause_btn = ActionButton("Pausar Bot")
        self.pause_btn.bind(on_release=lambda *_: self._run_action(self.client.pause))
        self.resume_btn = ActionButton("Reanudar Bot")
        self.resume_btn.bind(on_release=lambda *_: self._run_action(self.client.resume))

        self.add_widget(self.header)
        self.add_widget(self.status)
        self.add_widget(self.balance)
        self.add_widget(self.pnl)
        self.add_widget(self.positions)
        self.add_widget(self.pause_btn)
        self.add_widget(self.resume_btn)

        Clock.schedule_interval(lambda *_: self.refresh(), REFRESH_INTERVAL_SECONDS)
        self.refresh()

    def _run_action(self, action) -> None:
        result = action()
        if result.get("error"):
            self.header.text = f"Reco Trading Control - ERROR {result['error']}"
        else:
            self.header.text = "Reco Trading Control - OK"

    def refresh(self) -> None:
        health = self.client.health()
        metrics = self.client.metrics()
        positions = self.client.positions()

        if health.get("error"):
            self.status.set_value(f"OFFLINE ({health['error']})")
            return

        self.status.set_value(str(health.get("bot_status", "UNKNOWN")))
        self.balance.set_value(f"{metrics.get('balance', 0)} USDT")
        self.pnl.set_value(str(metrics.get("daily_pnl", 0)))
        self.positions.set_value("OPEN" if positions.get("has_open_position") else "NONE")
