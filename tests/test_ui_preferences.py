from __future__ import annotations

import json
from pathlib import Path

import pytest

from reco_trading.ui import preferences


def test_ui_preferences_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "prefs.json"
    monkeypatch.setattr(preferences, "_preferences_path", lambda: target)

    preferences.save_ui_preferences(theme="Light", language="Español")
    loaded = preferences.load_ui_preferences()

    assert loaded["theme"] == "Light"
    assert loaded["language"] == "Español"
    assert json.loads(target.read_text(encoding="utf-8"))["theme"] == "Light"


def test_ui_preferences_invalid_values_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "prefs.json"
    target.write_text(json.dumps({"theme": "Neon", "language": "German"}), encoding="utf-8")
    monkeypatch.setattr(preferences, "_preferences_path", lambda: target)

    loaded = preferences.load_ui_preferences()

    assert loaded["theme"] == "Dark"
    assert loaded["language"] == "English"
