from __future__ import annotations

from reco_trading import ui
from reco_trading.ui import theme as ui_theme


def test_ui_theme_backcompat_injects_get_theme_colors_if_missing(monkeypatch) -> None:
    monkeypatch.delattr(ui_theme, "get_theme_colors", raising=False)

    ui._ensure_theme_backcompat()

    assert hasattr(ui_theme, "get_theme_colors")
    palette = ui_theme.get_theme_colors("Dark")
    assert isinstance(palette, dict)
    assert "text_primary" in palette
