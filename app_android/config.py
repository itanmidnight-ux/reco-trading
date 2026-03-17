from __future__ import annotations

import os

AUTO_DISCOVERY = os.getenv("RECO_AUTO_DISCOVERY", "true").strip().lower() in {"1", "true", "yes", "on"}
API_URL = os.getenv("RECO_API_URL", "").strip().rstrip("/")
API_URL_CANDIDATES = tuple(
    url.strip().rstrip("/")
    for url in os.getenv(
        "RECO_API_URL_CANDIDATES",
        "http://10.0.2.2:8000,http://127.0.0.1:8000,http://localhost:8000",
    ).split(",")
    if url.strip()
)
BOOTSTRAP_URL = os.getenv("RECO_BOOTSTRAP_URL", "http://10.0.2.2:8000").strip().rstrip("/")
PUBLIC_API_URL = os.getenv("PUBLIC_API_URL", "").strip().rstrip("/")
API_KEY = os.getenv("RECO_API_KEY", "change-me")
REQUEST_TIMEOUT_SECONDS = int(os.getenv("RECO_REQUEST_TIMEOUT_SECONDS", "10"))
REFRESH_INTERVAL_SECONDS = float(os.getenv("RECO_REFRESH_INTERVAL_SECONDS", "2"))
CONNECT_RETRY_INTERVAL_SECONDS = float(os.getenv("RECO_CONNECT_RETRY_INTERVAL_SECONDS", "2"))
DISCOVERY_MAX_RETRIES = int(os.getenv("RECO_DISCOVERY_MAX_RETRIES", "4"))
DISCOVERY_BACKOFF_BASE_SECONDS = float(os.getenv("RECO_DISCOVERY_BACKOFF_BASE_SECONDS", "1"))
DISCOVERY_BACKOFF_MAX_SECONDS = float(os.getenv("RECO_DISCOVERY_BACKOFF_MAX_SECONDS", "8"))

REQUEST_RETRY_TOTAL = int(os.getenv("RECO_REQUEST_RETRY_TOTAL", "4"))
REQUEST_BACKOFF_FACTOR = float(os.getenv("RECO_REQUEST_BACKOFF_FACTOR", "0.4"))
