from __future__ import annotations

from reco_trading.ui.theme import app_stylesheet


def test_light_theme_uses_dark_text_for_readability() -> None:
    css = app_stylesheet(theme="Light")
    assert "#12213f" in css
    assert "color: #12213f" in css


def test_dark_theme_uses_light_text_for_readability() -> None:
    css = app_stylesheet(theme="Dark")
    assert "#edf2ff" in css
    assert "color: #edf2ff" in css
