from __future__ import annotations

from typing import Any

import requests

from config import API_KEY, API_URL, API_URL_CANDIDATES, REQUEST_TIMEOUT_SECONDS


class APIClient:
    def __init__(self) -> None:
        self.base_url = API_URL.rstrip("/")
        self.url_candidates = tuple(dict.fromkeys((self.base_url, *API_URL_CANDIDATES)))
        self.headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        self._session = requests.Session()

    def set_base_url(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

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

    def detect_reachable_base_url(self) -> str | None:
        original = self.base_url
        for candidate in self.url_candidates:
            self.set_base_url(candidate)
            result = self._request("GET", "/health", retries=1, timeout=min(3, REQUEST_TIMEOUT_SECONDS))
            if not result.get("error"):
                return candidate
        self.set_base_url(original)
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
