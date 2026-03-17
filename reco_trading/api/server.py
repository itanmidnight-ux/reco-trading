from __future__ import annotations

import os
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel

from reco_trading.config.settings import Settings
from reco_trading.core.runtime_control import RuntimeControl


class ClosePositionPayload(BaseModel):
    symbol: str


def _auth_guard(expected_key: str, authorization: str | None = Header(default=None)) -> None:
    if not expected_key:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="API auth key is not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != expected_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid bearer token")


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

    def require_auth(authorization: str | None = Header(default=None)) -> None:
        _auth_guard(os.getenv("API_AUTH_KEY", ""), authorization)

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

    @app.post("/close-position", dependencies=[Depends(require_auth)])
    async def close_position(payload: ClosePositionPayload) -> dict[str, Any]:
        state = runtime_control.snapshot()
        bot_symbol = str(state.get("snapshot", {}).get("pair", ""))
        if bot_symbol and bot_symbol != payload.symbol:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Bot is running symbol {bot_symbol}")
        runtime_control.enqueue("force_close")
        return {"accepted": True, "action": "force_close", "symbol": payload.symbol}

    @app.post("/pause", dependencies=[Depends(require_auth)])
    async def pause() -> dict[str, Any]:
        runtime_control.enqueue("pause")
        return {"accepted": True, "action": "pause"}

    @app.post("/resume", dependencies=[Depends(require_auth)])
    async def resume() -> dict[str, Any]:
        runtime_control.enqueue("resume")
        return {"accepted": True, "action": "resume"}

    @app.post("/kill-switch", dependencies=[Depends(require_auth)])
    async def kill_switch() -> dict[str, Any]:
        runtime_control.enqueue("kill_switch")
        return {"accepted": True, "action": "kill_switch"}

    return app
