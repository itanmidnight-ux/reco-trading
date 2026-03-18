from __future__ import annotations

import pytest
from fastapi import HTTPException

from reco_trading.api.server import _auth_guard, _password_digest, create_app
from reco_trading.config.settings import Settings
from reco_trading.core.runtime_control import RuntimeControl


def test_auth_guard_rejects_wrong_token() -> None:
    with pytest.raises(HTTPException) as exc:
        _auth_guard("expected-token", "Bearer wrong-token")
    assert exc.value.status_code == 401


def test_create_app_blocks_default_dashboard_credentials_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WEB_DASHBOARD_USER", raising=False)
    monkeypatch.delenv("WEB_DASHBOARD_PASSWORD", raising=False)
    monkeypatch.delenv("WEB_DASHBOARD_PASSWORD_SHA256", raising=False)
    settings = Settings.model_construct(runtime_profile="production")
    with pytest.raises(RuntimeError, match="must be changed in production"):
        create_app(RuntimeControl(), settings=settings)


def test_create_app_allows_hardened_dashboard_credentials_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEB_DASHBOARD_USER", "ops")
    monkeypatch.setenv("WEB_DASHBOARD_PASSWORD_SHA256", _password_digest("s3cure-pass"))
    settings = Settings.model_construct(runtime_profile="production")
    app = create_app(RuntimeControl(), settings=settings)
    assert app is not None


def test_readyz_route_is_registered() -> None:
    app = create_app(RuntimeControl(), settings=Settings.model_construct(runtime_profile="paper"))
    routes = {getattr(route, "path", "") for route in app.routes}
    assert "/readyz" in routes
