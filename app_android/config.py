from __future__ import annotations

import os


API_URL = os.getenv("RECO_API_URL", "http://10.0.2.2:8000")
API_URL_CANDIDATES = tuple(
    url.strip().rstrip("/")
    for url in os.getenv(
        "RECO_API_URL_CANDIDATES",
        "http://10.0.2.2:8000,http://192.168.1.100:8000,http://127.0.0.1:8000",
    ).split(",")
    if url.strip()
)
API_KEY = os.getenv("RECO_API_KEY", "change-me")
REQUEST_TIMEOUT_SECONDS = int(os.getenv("RECO_REQUEST_TIMEOUT_SECONDS", "8"))
REFRESH_INTERVAL_SECONDS = int(os.getenv("RECO_REFRESH_INTERVAL_SECONDS", "5"))
CONNECT_RETRY_INTERVAL_SECONDS = int(os.getenv("RECO_CONNECT_RETRY_INTERVAL_SECONDS", "3"))
