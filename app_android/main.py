from __future__ import annotations

from kivy.app import App
from kivy.core.window import Window

from ui.dashboard import Dashboard


class RecoTradingApp(App):
    def build(self):
        Window.clearcolor = (0.06, 0.08, 0.13, 1)
        return Dashboard()


if __name__ == "__main__":
    RecoTradingApp().run()
