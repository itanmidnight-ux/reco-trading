from __future__ import annotations

from kivy.metrics import dp
from kivy.properties import BooleanProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label


class Card(BoxLayout):
    def __init__(self, title: str, value: str = "-", **kwargs) -> None:
        super().__init__(orientation="vertical", padding=dp(14), spacing=dp(8), size_hint_y=None, height=dp(120), **kwargs)
        self.title_label = Label(text=title, font_size="14sp", color=(0.6, 0.7, 0.8, 1), halign="left")
        self.value_label = Label(text=value, font_size="28sp", bold=True, color=(0.95, 0.98, 1, 1), halign="left")
        self.add_widget(self.title_label)
        self.add_widget(self.value_label)
        self.canvas.before.clear()

    def set_value(self, value: str) -> None:
        self.value_label.text = value


class ActionButton(Button):
    """Standalone-safe action-like button.

    Kivy's default style rules may reference ``inside_group`` when widgets are
    treated like ActionBar controls. We define it explicitly so this widget can
    be used safely outside ActionBar contexts.
    """

    inside_group = BooleanProperty(False)

    def __init__(self, text: str, **kwargs) -> None:
        kwargs.setdefault("size_hint_y", None)
        kwargs.setdefault("height", dp(48))
        super().__init__(
            text=text,
            background_normal="",
            background_color=(0.10, 0.52, 0.98, 1),
            color=(1, 1, 1, 1),
            bold=True,
            **kwargs,
        )
        # Keep the explicit value to avoid implicit ActionBar context assumptions.
        self.inside_group = False
