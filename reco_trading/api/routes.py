"""
API Routes for Reco-Trading.
Additional REST endpoints for the API server.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel


logger = logging.getLogger(__name__)

router = APIRouter()
_context_provider: Any = None


def set_context_provider(provider: Any) -> None:
    """Register runtime context provider (bot/server) for real API data."""
    global _context_provider
    _context_provider = provider


def _resolve_snapshot() -> dict[str, Any]:
    provider = _context_provider
    if provider is None:
        return {}
    try:
        if callable(provider):
            provider = provider()
    except Exception:
        return {}
    snapshot = getattr(provider, "snapshot", {})
    if callable(snapshot):
        try:
            snapshot = snapshot()
        except Exception:
            snapshot = {}
    return snapshot if isinstance(snapshot, dict) else {}


class HealthResponse(BaseModel):
    status: str
    timestamp: str


class StatusResponse(BaseModel):
    bot_running: bool
    mode: str
    strategy: str
    timestamp: str


class BalanceResponse(BaseModel):
    total: float
    free: float
    locked: float
    timestamp: str


class TradeResponse(BaseModel):
    id: str
    pair: str
    side: str
    amount: float
    entry_price: float
    exit_price: float | None
    profit: float | None
    status: str
    timestamp: str


class OpenTradeResponse(BaseModel):
    id: str
    pair: str
    side: str
    amount: float
    entry_price: float
    current_price: float
    profit: float
    timestamp: str


class StatsResponse(BaseModel):
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_profit: float
    avg_profit: float
    timestamp: str


class ConfigResponse(BaseModel):
    strategy: str
    timeframe: str
    dry_run: bool
    max_open_trades: int
    stake_amount: float


class LogResponse(BaseModel):
    level: str
    message: str
    timestamp: str


class ForceEntryRequest(BaseModel):
    pair: str
    side: str = "long"
    amount: float | None = None


class ForceExitRequest(BaseModel):
    trade_id: str


@router.get("/", tags=["root"])
async def root():
    """Root endpoint."""
    return {
        "name": "Reco-Trading API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/api/v1/status", response_model=StatusResponse, tags=["bot"])
async def get_status():
    """Get bot status."""
    return {
        "bot_running": True,
        "mode": "dry_run",
        "strategy": "DefaultStrategy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/api/v1/balance", response_model=BalanceResponse, tags=["account"])
async def get_balance():
    """Get account balance."""
    snapshot = _resolve_snapshot()
    total = float(
        snapshot.get(
            "total_equity_usdt",
            snapshot.get("total_equity", snapshot.get("equity", snapshot.get("balance", 0.0))),
        )
        or 0.0
    )
    free = float(snapshot.get("operable_capital_usdt", snapshot.get("balance", 0.0)) or 0.0)
    locked = max(total - free, 0.0)
    return {
        "total": total,
        "free": free,
        "locked": locked,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/api/v1/trades", tags=["trades"])
async def get_trades(limit: int = 50, offset: int = 0):
    """Get trade history."""
    return {
        "trades": [],
        "count": 0,
        "limit": limit,
        "offset": offset,
    }


@router.get("/api/v1/trades/{trade_id}", tags=["trades"])
async def get_trade(trade_id: str):
    """Get specific trade."""
    return {
        "id": trade_id,
        "pair": "BTC/USDT",
        "side": "buy",
        "amount": 0.0,
        "entry_price": 0.0,
        "exit_price": None,
        "profit": None,
        "status": "open",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/api/v1/open-trades", tags=["trades"])
async def get_open_trades():
    """Get open trades."""
    return {
        "trades": [],
        "count": 0,
    }


@router.get("/api/v1/stats", response_model=StatsResponse, tags=["stats"])
async def get_stats():
    """Get trading statistics."""
    return {
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "win_rate": 0.0,
        "total_profit": 0.0,
        "avg_profit": 0.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/api/v1/config", response_model=ConfigResponse, tags=["config"])
async def get_config():
    """Get bot configuration."""
    return {
        "strategy": "DefaultStrategy",
        "timeframe": "5m",
        "dry_run": True,
        "max_open_trades": 3,
        "stake_amount": 0.05,
    }


@router.get("/api/v1/logs", response_model=list[LogResponse], tags=["logs"])
async def get_logs(level: str = "info", limit: int = 100):
    """Get recent logs."""
    return []


@router.post("/api/v1/forceentry", tags=["bot"])
async def force_entry(request: ForceEntryRequest):
    """Force entry for a trade."""
    return {
        "success": True,
        "message": "Trade entry queued",
        "pair": request.pair,
    }


@router.post("/api/v1/forceexit", tags=["bot"])
async def force_exit(request: ForceExitRequest):
    """Force exit for a trade."""
    return {
        "success": True,
        "message": "Trade exit queued",
        "trade_id": request.trade_id,
    }


@router.post("/api/v1/reload-config", tags=["bot"])
async def reload_config():
    """Reload bot configuration."""
    return {
        "success": True,
        "message": "Configuration reloaded",
    }


@router.get("/api/v1/pairs", tags=["market"])
async def get_pairs():
    """Get available trading pairs."""
    return {
        "whitelist": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        "blacklist": [],
    }


@router.get("/api/v1/pairs/{pair}/candles", tags=["market"])
async def get_candles(
    pair: str,
    timeframe: str = "5m",
    limit: int = 100
):
    """Get candles for a pair."""
    return {
        "pair": pair,
        "timeframe": timeframe,
        "candles": [],
        "count": 0,
    }


@router.get("/api/v1/pairs/{pair}/ticker", tags=["market"])
async def get_ticker(pair: str):
    """Get ticker for a pair."""
    return {
        "pair": pair,
        "last": 0.0,
        "bid": 0.0,
        "ask": 0.0,
        "volume": 0.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


__all__ = ["router"]
