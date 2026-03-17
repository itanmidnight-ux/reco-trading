from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _load_runtime_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parent / "runtime_config.json"
    if not config_path.exists():
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


_RUNTIME_CONFIG = _load_runtime_config()


def _value(name: str, default: str = "") -> str:
    if name in _RUNTIME_CONFIG and _RUNTIME_CONFIG[name] is not None:
        return str(_RUNTIME_CONFIG[name]).strip()
    return os.getenv(name, default).strip()


AUTO_DISCOVERY = _value("RECO_AUTO_DISCOVERY", "true").lower() in {"1", "true", "yes", "on"}
API_URL = _value("RECO_API_URL", "").rstrip("/")

_candidates_raw = _RUNTIME_CONFIG.get("RECO_API_URL_CANDIDATES")
if isinstance(_candidates_raw, list):
    _raw_candidates_text = ",".join(str(item) for item in _candidates_raw)
else:
    _raw_candidates_text = _value(
        "RECO_API_URL_CANDIDATES",
        "http://10.0.2.2:8000,http://127.0.0.1:8000,http://localhost:8000",
    )

API_URL_CANDIDATES = tuple(
    url.strip().rstrip("/")
    for url in _raw_candidates_text.split(",")
    if url.strip()
)

BOOTSTRAP_URL = _value("RECO_BOOTSTRAP_URL", "http://10.0.2.2:8000").rstrip("/")
PUBLIC_API_URL = _value("PUBLIC_API_URL", "").rstrip("/")
API_KEY = _value("RECO_API_KEY", "change-me")

REQUEST_TIMEOUT_SECONDS = int(_value("RECO_REQUEST_TIMEOUT_SECONDS", "10"))
REFRESH_INTERVAL_SECONDS = float(_value("RECO_REFRESH_INTERVAL_SECONDS", "2"))
CONNECT_RETRY_INTERVAL_SECONDS = float(_value("RECO_CONNECT_RETRY_INTERVAL_SECONDS", "2"))
DISCOVERY_MAX_RETRIES = int(_value("RECO_DISCOVERY_MAX_RETRIES", "4"))
DISCOVERY_BACKOFF_BASE_SECONDS = float(_value("RECO_DISCOVERY_BACKOFF_BASE_SECONDS", "1"))
DISCOVERY_BACKOFF_MAX_SECONDS = float(_value("RECO_DISCOVERY_BACKOFF_MAX_SECONDS", "8"))
REQUEST_RETRY_TOTAL = int(_value("RECO_REQUEST_RETRY_TOTAL", "4"))
REQUEST_BACKOFF_FACTOR = float(_value("RECO_REQUEST_BACKOFF_FACTOR", "0.4"))
