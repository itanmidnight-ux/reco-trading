from __future__ import annotations

from typing import Any

import requests

from config import API_KEY, API_URL, REQUEST_TIMEOUT_SECONDS


class APIClient:
    def __init__(self) -> None:
        self.base_url = API_URL.rstrip("/")
        self.headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    def _request(self, method: str, path: str, json: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            response = requests.request(
                method=method,
                url=f"{self.base_url}{path}",
                headers=self.headers,
                json=json,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return response.json()
        except requests.Timeout:
            return {"error": "timeout"}
        except requests.RequestException as exc:
            return {"error": str(exc)}

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def metrics(self) -> dict[str, Any]:
        return self._request("GET", "/metrics")

    def positions(self) -> dict[str, Any]:
        return self._request("GET", "/positions")

    def pause(self) -> dict[str, Any]:
        return self._request("POST", "/pause")

    def resume(self) -> dict[str, Any]:
        return self._request("POST", "/resume")

    def close_position(self, symbol: str) -> dict[str, Any]:
        return self._request("POST", "/close-position", json={"symbol": symbol})
