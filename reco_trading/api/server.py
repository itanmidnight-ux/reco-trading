from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from reco_trading.config.settings import Settings
from reco_trading.core.runtime_control import RuntimeControl


class ClosePositionPayload(BaseModel):
    symbol: str


class RuntimeSettingsPayload(BaseModel):
    investment_mode: str = "Balanced"
    risk_per_trade_fraction: float = Field(default=0.01, ge=0.001, le=0.1)
    max_trade_balance_fraction: float = Field(default=0.20, ge=0.01, le=1.0)
    capital_limit_usdt: float = Field(default=0.0, ge=0.0)
    symbol_capital_limits: dict[str, float] = Field(default_factory=dict)


class LoginPayload(BaseModel):
    username: str
    password: str


LOGIN_ATTEMPTS: dict[str, list[float]] = {}
DASHBOARD_SESSIONS: dict[str, dict[str, Any]] = {}
MAX_LOGIN_ATTEMPTS = 5
LOGIN_WINDOW_SECONDS = 600
SESSION_TTL_SECONDS = 8 * 60 * 60
SESSION_COOKIE_NAME = "dashboard_session"


def _password_digest(raw_password: str) -> str:
    return hashlib.sha256(raw_password.encode("utf-8")).hexdigest()


def _is_login_rate_limited(client_host: str) -> bool:
    now = time.time()
    attempts = [ts for ts in LOGIN_ATTEMPTS.get(client_host, []) if now - ts <= LOGIN_WINDOW_SECONDS]
    LOGIN_ATTEMPTS[client_host] = attempts
    return len(attempts) >= MAX_LOGIN_ATTEMPTS


def _register_failed_login(client_host: str) -> None:
    LOGIN_ATTEMPTS.setdefault(client_host, []).append(time.time())


def _auth_guard(expected_key: str, authorization: str | None = Header(default=None)) -> None:
    if not expected_key:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="API auth key is not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != expected_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid bearer token")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        response = await call_next(request)
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "font-src 'self' data:; connect-src 'self'; frame-ancestors 'none';"
        )
        return response


def _safe_settings_payload(settings: Settings | None) -> dict[str, Any]:
    if settings is None:
        return {}
    return {
        "environment": settings.environment,
        "runtime_profile": settings.runtime_profile,
        "symbol": settings.symbol,
        "timeframe": f"{settings.primary_timeframe}/{settings.confirmation_timeframe}",
        "risk_per_trade_fraction": settings.risk_per_trade_fraction,
        "max_trade_balance_fraction": settings.max_trade_balance_fraction,
        "daily_loss_limit_fraction": settings.daily_loss_limit_fraction,
        "max_drawdown_fraction": settings.max_drawdown_fraction,
    }


def create_app(runtime_control: RuntimeControl, settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="reco_trading control API", version="1.0.0")
    app.add_middleware(SecurityHeadersMiddleware)

    web_root = Path(__file__).resolve().parents[1] / "web"
    app.mount("/dashboard/static", StaticFiles(directory=web_root / "static"), name="dashboard-static")

    dashboard_username = os.getenv("WEB_DASHBOARD_USER", "fabian")
    dashboard_password = os.getenv("WEB_DASHBOARD_PASSWORD", "admin123")
    expected_password_digest = os.getenv("WEB_DASHBOARD_PASSWORD_SHA256", _password_digest(dashboard_password))
    secure_cookie = os.getenv("WEB_SECURE_COOKIE", "").lower() in {"1", "true", "yes"}

    def require_auth(authorization: str | None = Header(default=None)) -> None:
        _auth_guard(os.getenv("API_AUTH_KEY", ""), authorization)

    def _load_session(request: Request) -> dict[str, Any]:
        token = str(request.cookies.get(SESSION_COOKIE_NAME, "")).strip()
        if not token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="dashboard_not_authenticated")
        session = DASHBOARD_SESSIONS.get(token)
        if not session:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="dashboard_not_authenticated")
        if float(session.get("expires_at", 0)) < time.time():
            DASHBOARD_SESSIONS.pop(token, None)
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="dashboard_session_expired")
        return session

    def require_dashboard_auth(request: Request) -> str:
        session = _load_session(request)
        username = str(session.get("username", "")).strip()
        if username != dashboard_username:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="dashboard_not_authenticated")
        return username

    def require_csrf(request: Request) -> None:
        token = str(request.headers.get("X-CSRF-Token", "")).strip()
        session_token = str(_load_session(request).get("csrf_token", "")).strip()
        if not token or not session_token or not hmac.compare_digest(token, session_token):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="invalid_csrf_token")

    def authorize_control_access(request: Request, authorization: str | None = Header(default=None)) -> None:
        if authorization and authorization.startswith("Bearer "):
            _auth_guard(os.getenv("API_AUTH_KEY", ""), authorization)
            return
        require_dashboard_auth(request)
        require_csrf(request)

    async def _dashboard_payload() -> dict[str, Any]:
        state = runtime_control.snapshot()
        snapshot = state.get("snapshot", {})
        balance = snapshot.get("balance")
        equity = snapshot.get("equity", snapshot.get("total_equity"))
        daily_pnl = snapshot.get("daily_pnl")
        win_rate = snapshot.get("win_rate")
        return {
            "health": {
                "status": "ok",
                "uptime_seconds": state["uptime_seconds"],
                "bot_status": state["bot_status"],
                "open_positions": state["open_positions"],
                "restart_count": state["restart_count"],
                "heartbeat_age_seconds": state["heartbeat_age_seconds"],
            },
            "metrics": {
                "balance": balance,
                "equity": equity,
                "total_equity": snapshot.get("total_equity", equity),
                "daily_pnl": daily_pnl,
                "session_pnl": snapshot.get("session_pnl"),
                "win_rate": win_rate,
                "trades_today": snapshot.get("trades_today"),
                "btc_balance": snapshot.get("btc_balance"),
                "btc_value": snapshot.get("btc_value"),
                "price": snapshot.get("price"),
                "bid": snapshot.get("bid"),
                "ask": snapshot.get("ask"),
                "spread": snapshot.get("spread"),
                "signal": snapshot.get("signal"),
                "confidence": snapshot.get("confidence"),
                "status": snapshot.get("status"),
            },
            "positions": {
                "open_position": snapshot.get("open_position"),
                "has_open_position": snapshot.get("has_open_position"),
                "symbol": snapshot.get("pair"),
            },
            "runtime": state,
            "snapshot": snapshot,
            "settings": _safe_settings_payload(settings),
        }

    @app.get("/", response_class=HTMLResponse)
    async def dashboard_index() -> str:
        return (web_root / "templates" / "dashboard.html").read_text(encoding="utf-8")

    @app.post("/dashboard/login")
    async def dashboard_login(payload: LoginPayload, request: Request, response: Response) -> dict[str, str]:
        client_host = request.client.host if request.client else "unknown"
        if _is_login_rate_limited(client_host):
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="too_many_login_attempts")

        user_ok = hmac.compare_digest(payload.username, dashboard_username)
        pass_ok = hmac.compare_digest(_password_digest(payload.password), expected_password_digest)
        if not (user_ok and pass_ok):
            _register_failed_login(client_host)
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_credentials")

        session_token = secrets.token_urlsafe(48)
        csrf_token = secrets.token_urlsafe(32)
        DASHBOARD_SESSIONS[session_token] = {
            "username": payload.username,
            "csrf_token": csrf_token,
            "expires_at": time.time() + SESSION_TTL_SECONDS,
        }
        response.set_cookie(
            SESSION_COOKIE_NAME,
            session_token,
            httponly=True,
            secure=secure_cookie,
            samesite="strict",
            max_age=SESSION_TTL_SECONDS,
        )
        return {"ok": "true", "csrf_token": csrf_token}

    @app.post("/dashboard/logout")
    async def dashboard_logout(request: Request, response: Response, username: str = Depends(require_dashboard_auth)) -> dict[str, str]:
        _ = username
        token = str(request.cookies.get(SESSION_COOKIE_NAME, "")).strip()
        if token:
            DASHBOARD_SESSIONS.pop(token, None)
        response.delete_cookie(SESSION_COOKIE_NAME)
        return {"ok": "true"}

    @app.get("/dashboard/session")
    async def dashboard_session(request: Request, username: str = Depends(require_dashboard_auth)) -> dict[str, str]:
        session = _load_session(request)
        return {"username": username, "csrf_token": str(session.get("csrf_token", ""))}

    @app.get("/dashboard/data")
    async def dashboard_data(username: str = Depends(require_dashboard_auth)) -> dict[str, Any]:
        _ = username
        return await _dashboard_payload()

    @app.get("/health", dependencies=[Depends(require_auth)])
    async def health() -> dict[str, Any]:
        state = runtime_control.snapshot()
        return {
            "status": "ok",
            "uptime_seconds": state["uptime_seconds"],
            "bot_status": state["bot_status"],
            "open_positions": state["open_positions"],
            "restart_count": state["restart_count"],
            "heartbeat_age_seconds": state["heartbeat_age_seconds"],
        }

    @app.get("/metrics", dependencies=[Depends(require_auth)])
    async def metrics() -> dict[str, Any]:
        state = runtime_control.snapshot()
        snapshot = state.get("snapshot", {})
        return {
            "balance": snapshot.get("balance"),
            "equity": snapshot.get("equity"),
            "daily_pnl": snapshot.get("daily_pnl"),
            "win_rate": snapshot.get("win_rate"),
            "status": snapshot.get("status"),
        }

    @app.get("/positions", dependencies=[Depends(require_auth)])
    async def positions() -> dict[str, Any]:
        state = runtime_control.snapshot()
        snapshot = state.get("snapshot", {})
        return {
            "open_position": snapshot.get("open_position"),
            "has_open_position": snapshot.get("has_open_position"),
            "symbol": snapshot.get("pair"),
        }

    @app.get("/runtime", dependencies=[Depends(require_auth)])
    async def runtime() -> dict[str, Any]:
        return runtime_control.snapshot()

    @app.get("/settings", dependencies=[Depends(require_auth)])
    async def runtime_settings() -> dict[str, Any]:
        return _safe_settings_payload(settings)

    @app.get("/public-url")
    async def public_url() -> dict[str, Any]:
        url = os.getenv("PUBLIC_API_URL", "").strip()
        if url and not url.startswith("https://"):
            return {"url": None, "error": "invalid_public_url"}
        return {"url": url or None}

    @app.post("/runtime-settings", dependencies=[Depends(authorize_control_access)])
    async def apply_runtime_settings(payload: RuntimeSettingsPayload) -> dict[str, Any]:
        runtime_control.enqueue("runtime_settings", **payload.model_dump())
        return {"accepted": True, "action": "runtime_settings", "payload": payload.model_dump()}

    @app.post("/close-position", dependencies=[Depends(authorize_control_access)])
    async def close_position(payload: ClosePositionPayload) -> dict[str, Any]:
        state = runtime_control.snapshot()
        bot_symbol = str(state.get("snapshot", {}).get("pair", ""))
        if bot_symbol and bot_symbol != payload.symbol:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Bot is running symbol {bot_symbol}")
        runtime_control.enqueue("force_close")
        return {"accepted": True, "action": "force_close", "symbol": payload.symbol}

    @app.post("/start", dependencies=[Depends(authorize_control_access)])
    async def start() -> dict[str, Any]:
        runtime_control.enqueue("resume")
        return {"accepted": True, "action": "start"}

    @app.post("/pause", dependencies=[Depends(authorize_control_access)])
    async def pause() -> dict[str, Any]:
        runtime_control.enqueue("pause")
        return {"accepted": True, "action": "pause"}

    @app.post("/resume", dependencies=[Depends(authorize_control_access)])
    async def resume() -> dict[str, Any]:
        runtime_control.enqueue("resume")
        return {"accepted": True, "action": "resume"}

    @app.post("/kill-switch", dependencies=[Depends(authorize_control_access)])
    async def kill_switch() -> dict[str, Any]:
        runtime_control.enqueue("kill_switch")
        return {"accepted": True, "action": "kill_switch"}

    return app
