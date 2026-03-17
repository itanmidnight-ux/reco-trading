from __future__ import annotations

from kivy.animation import Animation
from kivy.clock import Clock
from kivy.graphics import Color, RoundedRectangle
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput


class GradientSection(BoxLayout):
    def __init__(self, **kwargs) -> None:
        super().__init__(orientation="vertical", padding=dp(12), spacing=dp(8), size_hint_y=None, **kwargs)
        self.bind(minimum_height=self.setter("height"), pos=self._redraw, size=self._redraw)
        with self.canvas.before:
            self._bg_color = Color(0.11, 0.15, 0.23, 0.95)
            self._bg_rect = RoundedRectangle(radius=[dp(18)] * 4)
        self._pulse: Animation | None = None

    def _redraw(self, *_args) -> None:
        self._bg_rect.pos = self.pos
        self._bg_rect.size = self.size

    def pulse(self) -> None:
        if self._pulse is not None:
            self._pulse.cancel(self)
        self._pulse = Animation(opacity=0.92, duration=0.25) + Animation(opacity=1, duration=0.25)
        self._pulse.start(self)


class Card(BoxLayout):
    def __init__(self, title: str, value: str = "-", subtitle: str = "", **kwargs) -> None:
        super().__init__(orientation="vertical", padding=dp(14), spacing=dp(4), size_hint_y=None, height=dp(108), **kwargs)
        self.bind(pos=self._redraw, size=self._redraw)
        with self.canvas.before:
            self._bg = Color(0.14, 0.20, 0.31, 0.92)
            self._bg_rect = RoundedRectangle(radius=[dp(14)] * 4)

        self.title_label = Label(text=title, font_size="13sp", color=(0.70, 0.78, 0.90, 1), halign="left", valign="middle")
        self.value_label = Label(text=value, font_size="20sp", bold=True, color=(0.95, 0.98, 1, 1), halign="left", valign="middle")
        self.subtitle_label = Label(text=subtitle, font_size="12sp", color=(0.62, 0.72, 0.85, 1), halign="left", valign="middle")
        self.add_widget(self.title_label)
        self.add_widget(self.value_label)
        self.add_widget(self.subtitle_label)
        Clock.schedule_once(lambda *_: self._sync_text(), 0)

    def _sync_text(self, *_args) -> None:
        for widget in (self.title_label, self.value_label, self.subtitle_label):
            widget.text_size = (self.width - dp(10), None)

    def _redraw(self, *_args) -> None:
        self._bg_rect.pos = self.pos
        self._bg_rect.size = self.size
        self._sync_text()

    def set_value(self, value: str, subtitle: str | None = None, color: tuple[float, float, float, float] | None = None) -> None:
        self.value_label.text = value
        if subtitle is not None:
            self.subtitle_label.text = subtitle
        if color is not None:
            self.value_label.color = color
        anim = Animation(opacity=0.8, duration=0.08) + Animation(opacity=1, duration=0.08)
        anim.start(self.value_label)


class PrimaryButton(Button):
    def __init__(self, text: str, tone: str = "primary", **kwargs) -> None:
        palette = {
            "primary": (0.17, 0.44, 0.94, 1),
            "positive": (0.07, 0.67, 0.45, 1),
            "danger": (0.85, 0.24, 0.27, 1),
            "warning": (0.92, 0.68, 0.15, 1),
        }
        super().__init__(
            text=text,
            size_hint_y=None,
            height=dp(48),
            background_normal="",
            background_color=palette.get(tone, palette["primary"]),
            color=(1, 1, 1, 1),
            bold=True,
            **kwargs,
        )


class LabeledField(BoxLayout):
    def __init__(self, label: str, value: str, readonly: bool = False, **kwargs) -> None:
        super().__init__(orientation="vertical", spacing=dp(4), size_hint_y=None, height=dp(76), **kwargs)
        self.caption = Label(text=label, font_size="12sp", color=(0.73, 0.79, 0.90, 1), halign="left")
        self.input = TextInput(
            text=value,
            multiline=False,
            readonly=readonly,
            background_normal="",
            background_active="",
            background_color=(0.09, 0.12, 0.2, 0.95),
            foreground_color=(0.95, 0.98, 1, 1),
            cursor_color=(0.72, 0.82, 1, 1),
            padding=[dp(10), dp(12), dp(10), dp(12)],
        )
        self.add_widget(self.caption)
        self.add_widget(self.input)
        Clock.schedule_once(lambda *_: self._sync_text(), 0)

    def _sync_text(self) -> None:
        self.caption.text_size = (self.width - dp(8), None)


class VerticalScroll(ScrollView):
    def __init__(self, **kwargs) -> None:
        super().__init__(do_scroll_x=False, bar_width=dp(4), scroll_type=["bars", "content"], **kwargs)
