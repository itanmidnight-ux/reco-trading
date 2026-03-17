from __future__ import annotations

import logging
import time
from typing import Any

import requests

from config import (
    API_KEY,
    API_URL,
    API_URL_CANDIDATES,
    AUTO_DISCOVERY,
    BOOTSTRAP_URL,
    DISCOVERY_BACKOFF_BASE_SECONDS,
    DISCOVERY_BACKOFF_MAX_SECONDS,
    DISCOVERY_MAX_RETRIES,
    PUBLIC_API_URL,
    REQUEST_TIMEOUT_SECONDS,
)


logger = logging.getLogger(__name__)


class APIClient:
    def __init__(self) -> None:
        self.base_url = API_URL.rstrip("/")
        self.headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        self._session = requests.Session()
        self._refresh_candidates()

    def _refresh_candidates(self) -> None:
        candidates = []
        if self.base_url:
            candidates.append(self.base_url)
        candidates.extend(API_URL_CANDIDATES)
        if BOOTSTRAP_URL:
            candidates.append(BOOTSTRAP_URL)
        if PUBLIC_API_URL:
            candidates.append(PUBLIC_API_URL)
        self.url_candidates = tuple(dict.fromkeys(url.rstrip("/") for url in candidates if url))

    def set_base_url(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._refresh_candidates()

    def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        retries: int = 1,
        timeout: int | float = REQUEST_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        last_error = "request_failed"
        for _ in range(max(1, retries)):
            try:
                response = self._session.request(
                    method=method,
                    url=f"{self.base_url}{path}",
                    headers=self.headers,
                    json=json,
                    timeout=timeout,
                )
                response.raise_for_status()
                if response.content:
                    return response.json()
                return {"ok": True}
            except requests.Timeout:
                last_error = "timeout"
            except requests.ConnectionError:
                last_error = "connection_error"
            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                last_error = f"http_error:{status_code}" if status_code else "http_error"
            except ValueError:
                last_error = "invalid_json"
            except requests.RequestException as exc:
                last_error = str(exc)
        return {"error": last_error}

    def _validate_candidate(self, candidate: str, timeout: int | float = 3) -> bool:
        self.set_base_url(candidate)
        result = self._request("GET", "/health", retries=1, timeout=timeout)
        return not result.get("error")

    def _discover_public_url_from_bootstrap(self) -> str | None:
        bootstrap = BOOTSTRAP_URL.rstrip("/")
        if not bootstrap:
            return None

        original = self.base_url
        self.set_base_url(bootstrap)
        payload = self._request("GET", "/public-url", retries=1, timeout=min(3, REQUEST_TIMEOUT_SECONDS))
        self.set_base_url(original)

        if payload.get("error"):
            return None

        discovered = str(payload.get("url") or "").strip().rstrip("/")
        if not discovered.startswith("https://"):
            return None
        return discovered

    def detect_reachable_base_url(self) -> str | None:
        self._refresh_candidates()

        for candidate in self.url_candidates:
            logger.info("Trying API candidate: %s", candidate)
            if self._validate_candidate(candidate):
                logger.info("Connected to API candidate: %s", candidate)
                return candidate

        if not AUTO_DISCOVERY:
            return None

        delay = max(DISCOVERY_BACKOFF_BASE_SECONDS, 0.2)
        for _ in range(max(1, DISCOVERY_MAX_RETRIES)):
            discovered = self._discover_public_url_from_bootstrap()
            if discovered:
                logger.info("Discovered public API URL from bootstrap: %s", discovered)
            if discovered and self._validate_candidate(discovered, timeout=REQUEST_TIMEOUT_SECONDS):
                logger.info("Connected to discovered API URL: %s", discovered)
                return discovered

            if PUBLIC_API_URL:
                logger.info("Trying PUBLIC_API_URL fallback: %s", PUBLIC_API_URL)
            if PUBLIC_API_URL and self._validate_candidate(PUBLIC_API_URL, timeout=REQUEST_TIMEOUT_SECONDS):
                logger.info("Connected via PUBLIC_API_URL fallback: %s", PUBLIC_API_URL)
                return PUBLIC_API_URL

            time.sleep(delay)
            delay = min(delay * 2, DISCOVERY_BACKOFF_MAX_SECONDS)

        return None

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health", retries=2)

    def metrics(self) -> dict[str, Any]:
        return self._request("GET", "/metrics", retries=2)

    def positions(self) -> dict[str, Any]:
        return self._request("GET", "/positions", retries=2)

    def pause(self) -> dict[str, Any]:
        return self._request("POST", "/pause", retries=2)

    def resume(self) -> dict[str, Any]:
        return self._request("POST", "/resume", retries=2)

    def close_position(self, symbol: str) -> dict[str, Any]:
        return self._request("POST", "/close-position", json={"symbol": symbol}, retries=2)
