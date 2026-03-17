from __future__ import annotations

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window

from config import CONNECT_RETRY_INTERVAL_SECONDS
from ui.dashboard import Dashboard


class RecoTradingApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.connected = False
        self.active_base_url = ""
        self.dashboard: Dashboard | None = None

    def build(self):
        Window.clearcolor = (0.06, 0.08, 0.13, 1)
        self.dashboard = Dashboard()
        return self.dashboard

    def on_start(self) -> None:
        Clock.schedule_once(lambda *_: self._attempt_connect(), 0)
        Clock.schedule_interval(lambda *_: self._attempt_connect(), CONNECT_RETRY_INTERVAL_SECONDS)

    def _attempt_connect(self) -> None:
        if self.dashboard is None:
            return

        candidate = self.dashboard.client.detect_reachable_base_url()
        if candidate:
            self.dashboard.client.set_base_url(candidate)
            self.active_base_url = candidate
            self.connected = True
            self.dashboard.set_connection(True, candidate)
            return

        self.connected = False
        self.dashboard.set_connection(False)


if __name__ == "__main__":
    RecoTradingApp().run()
