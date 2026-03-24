from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reco_trading.ui.state_manager import StateManager as StateManagerType


__all__ = ["run_gui", "StateManager"]


def _ensure_theme_backcompat() -> None:
    from reco_trading.ui import theme as _theme

    if hasattr(_theme, "get_theme_colors"):
        return

    def _fallback_get_theme_colors(theme: str = "Dark") -> dict[str, str]:
        normalized = str(theme or "Dark").strip().lower()
        if normalized in {"light", "white", "blanco"}:
            return {
                "background": "#f5f7fb",
                "background_alt": "#e8edf7",
                "panel": "#ffffff",
                "panel_alt": "#f1f5fc",
                "border": "#c7d3ea",
                "text_primary": "#12213f",
                "text_secondary": "#425a85",
                "positive": "#0f9f6e",
                "negative": "#d6405c",
                "warning": "#b7791f",
                "info": "#2c5bd8",
                "neutral": "#6f84b0",
                "accent": "#4b54e6",
            }
        return {
            "background": "#0b1020",
            "background_alt": "#111933",
            "panel": "#141d35",
            "panel_alt": "#1a2643",
            "border": "#273658",
            "text_primary": "#edf2ff",
            "text_secondary": "#9fb2d9",
            "positive": "#22d39b",
            "negative": "#ff5f7b",
            "warning": "#ffcc66",
            "info": "#5a8dff",
            "neutral": "#7f93bf",
            "accent": "#7b61ff",
        }

    setattr(_theme, "get_theme_colors", _fallback_get_theme_colors)


_ensure_theme_backcompat()


def run_gui(state_manager: "StateManagerType") -> int:
    from reco_trading.ui.app import run_gui as _run_gui

    return _run_gui(state_manager)


def StateManager():
    from reco_trading.ui.state_manager import StateManager as _StateManager

    return _StateManager()
