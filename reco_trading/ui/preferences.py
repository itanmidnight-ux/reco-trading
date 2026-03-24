from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reco_trading.ui.i18n import normalize_language


def _preferences_path() -> Path:
    return Path.home() / ".reco_trading" / "ui_preferences.json"


def load_ui_preferences() -> dict[str, Any]:
    path = _preferences_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"theme": "Dark", "language": "English"}
    if not isinstance(payload, dict):
        return {"theme": "Dark", "language": "English"}
    theme = str(payload.get("theme", "Dark")).strip()
    if theme not in {"Dark", "Light"}:
        theme = "Dark"
    language = normalize_language(str(payload.get("language", "English")))
    return {"theme": theme, "language": language}


def save_ui_preferences(*, theme: str, language: str) -> None:
    path = _preferences_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "theme": theme if theme in {"Dark", "Light"} else "Dark",
            "language": normalize_language(language),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return
