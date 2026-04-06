from __future__ import annotations

import asyncio
import base64
import hmac
import json
import logging
import os
import socket
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from flask import Flask, render_template, jsonify, Response, request

try:
    import jwt
except Exception:  # pragma: no cover - optional dependency
    jwt = None

try:
    from flask_sock import Sock
except Exception:  # pragma: no cover - optional dependency
    Sock = None

from reco_trading.database.repository import Repository

logger = logging.getLogger(__name__)

_global_bot_instance: Any = None
_bot_instance_getter: Callable | None = None
_fallback_repository: Repository | None = None
_app: Flask | None = None
_sock: Sock | None = None
_connected_clients: list[Any] = []
_connected_clients_lock = threading.Lock()

JWT_SECRET = os.getenv("DASHBOARD_JWT_SECRET", "your-secret-key-change-this")
JWT_EXPIRATION_HOURS = 24


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, str(default))).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _is_dashboard_auth_enabled() -> bool:
    # Auth habilitada por defecto para seguridad. Deshabilitar explícitamente con DASHBOARD_AUTH_ENABLED=false
    return _env_bool("DASHBOARD_AUTH_ENABLED", True)


def _dashboard_token() -> str:
    return str(os.getenv("DASHBOARD_API_TOKEN", "")).strip()


def _dashboard_user() -> str:
    return str(os.getenv("DASHBOARD_USERNAME", "admin")).strip()


def _dashboard_password() -> str:
    return str(os.getenv("DASHBOARD_PASSWORD", "")).strip()

def _create_jwt_token(username: str) -> str:
    if jwt is None:
        issued = int(datetime.now(timezone.utc).timestamp())
        payload = f"{username}:{issued}"
        signature = hmac.new(JWT_SECRET.encode("utf-8"), payload.encode("utf-8"), "sha256").hexdigest()
        return base64.urlsafe_b64encode(f"{payload}:{signature}".encode("utf-8")).decode("utf-8")
    payload = {
        "username": username,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def _verify_jwt_token(token: str) -> dict | None:
    if jwt is None:
        try:
            decoded = base64.urlsafe_b64decode(token.encode("utf-8")).decode("utf-8")
            username, issued_str, signature = decoded.split(":", 2)
            payload = f"{username}:{issued_str}"
            expected = hmac.new(JWT_SECRET.encode("utf-8"), payload.encode("utf-8"), "sha256").hexdigest()
            if not hmac.compare_digest(signature, expected):
                return None
            issued = datetime.fromtimestamp(int(issued_str), tz=timezone.utc)
            if datetime.now(timezone.utc) - issued > timedelta(hours=JWT_EXPIRATION_HOURS):
                return None
            return {"username": username, "iat": issued_str}
        except Exception:
            return None
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except Exception:
        return None


def _require_auth(f):
    from functools import wraps

    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
        if not token or not _verify_jwt_token(token):
            return _unauthorized_response()
        return f(*args, **kwargs)

    return decorated_function


def _unauthorized_response() -> Response:
    response = jsonify({"success": False, "error": "Unauthorized"})
    response.status_code = 401
    mode = str(os.getenv("DASHBOARD_AUTH_MODE", "hybrid")).strip().lower()
    if mode in {"basic", "hybrid"}:
        response.headers["WWW-Authenticate"] = 'Basic realm="Reco Dashboard"'
    return response


def _check_basic_auth() -> bool:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(auth_header.split(" ", 1)[1]).decode("utf-8")
        username, password = decoded.split(":", 1)
    except Exception:
        return False
    configured_user = _dashboard_user()
    configured_password = _dashboard_password()
    if not configured_user or not configured_password:
        return False
    return hmac.compare_digest(username, configured_user) and hmac.compare_digest(password, configured_password)


def _check_token_auth() -> bool:
    token = _dashboard_token()
    if not token:
        return False
    supplied = (
        request.headers.get("X-Dashboard-Token", "")
        or request.args.get("token", "")
    ).strip()
    auth_header = request.headers.get("Authorization", "")
    if not supplied and auth_header.startswith("Bearer "):
        supplied = auth_header.split(" ", 1)[1].strip()
    return bool(supplied) and hmac.compare_digest(supplied, token)


def _is_same_origin_request() -> bool:
    """Check if request comes from the same Flask app (same origin)."""
    origin = request.headers.get("Origin", "")
    referer = request.headers.get("Referer", "")
    host = request.host

    if origin:
        from urllib.parse import urlparse
        parsed = urlparse(origin)
        if parsed.netloc == host:
            return True
        # Also check host without port
        host_no_port = host.split(":")[0]
        origin_host = parsed.hostname or ""
        if origin_host == host_no_port:
            return True

    if referer:
        from urllib.parse import urlparse
        parsed = urlparse(referer)
        if parsed.netloc == host:
            return True
        # Also check host without port
        host_no_port = host.split(":")[0]
        referer_host = parsed.hostname or ""
        if referer_host == host_no_port:
            return True

    return False


def _is_authorized() -> bool:
    if not _is_dashboard_auth_enabled():
        return True
    # Same-origin requests from the dashboard UI are always authorized
    if _is_same_origin_request():
        return True
    # Compatibilidad: si no hay credenciales configuradas, no bloquear el dashboard.
    if not _dashboard_token() and not _dashboard_password():
        return True
    mode = str(os.getenv("DASHBOARD_AUTH_MODE", "hybrid")).strip().lower()
    if mode == "token":
        return _check_token_auth()
    if mode == "basic":
        return _check_basic_auth()
    return _check_token_auth() or _check_basic_auth()




def _validate_login_payload(payload: dict[str, Any]) -> tuple[bool, str]:
    if not _is_dashboard_auth_enabled():
        return True, "auth_disabled"

    provided_user = str(payload.get("user", "")).strip()
    provided_password = str(payload.get("password", "")).strip()
    provided_token = str(payload.get("token", "")).strip()

    configured_user = _dashboard_user()
    configured_password = _dashboard_password()
    configured_token = _dashboard_token()

    if configured_user and not (provided_user and hmac.compare_digest(provided_user, configured_user)):
        return False, "Invalid user"
    if configured_password and not (provided_password and hmac.compare_digest(provided_password, configured_password)):
        return False, "Invalid password"
    if configured_token and not (provided_token and hmac.compare_digest(provided_token, configured_token)):
        return False, "Invalid token"

    return True, "ok"

def _require_dashboard_auth() -> Response | None:
    if _is_authorized():
        return None
    return _unauthorized_response()


def _calc_duration(start: datetime | None, end: datetime | None) -> str | None:
    """Calculate trade duration as human-readable string."""
    if not start or not end:
        return None
    try:
        delta = end - start
        total_seconds = int(delta.total_seconds())
        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            return f"{total_seconds // 60}m"
        elif total_seconds < 86400:
            return f"{total_seconds // 3600}h {total_seconds % 3600 // 60}m"
        else:
            return f"{total_seconds // 86400}d {total_seconds % 86400 // 3600}h"
    except Exception:
        return None


def set_bot_instance(bot) -> None:
    """Set the global bot instance for the dashboard to access."""
    global _global_bot_instance
    _global_bot_instance = bot
    logger.info("Bot instance set for web dashboard")


def set_bot_instance_getter(getter: Callable) -> None:
    """Set a callable that returns the bot instance."""
    global _bot_instance_getter
    _bot_instance_getter = getter
    logger.info("Bot instance getter set for web dashboard")


# Loop persistente para operaciones de BD del dashboard (evita crear/destruir loops en cada SSE tick)
_dashboard_db_loop: asyncio.AbstractEventLoop | None = None
_dashboard_db_loop_lock = threading.Lock()


def _get_dashboard_db_loop() -> asyncio.AbstractEventLoop:
    global _dashboard_db_loop
    with _dashboard_db_loop_lock:
        if _dashboard_db_loop is None or _dashboard_db_loop.is_closed():
            _dashboard_db_loop = asyncio.new_event_loop()
            _dashboard_db_loop.set_debug(False)
        return _dashboard_db_loop


def _run_async(coro: Any) -> Any:
    """
    Ejecuta coroutines de forma segura desde contexto Flask (síncrono).
    Usa un loop persistente para evitar memory leaks en polling SSE.
    NUNCA llamar con coroutines que accedan a objetos internos del bot.
    """
    try:
        loop = _get_dashboard_db_loop()
        if loop.is_running():
            # No debería ocurrir en Flask sync, pero por seguridad:
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result(timeout=10.0)
        return loop.run_until_complete(coro)
    except Exception as e:
        logger.error("_run_async error: %s", e)
        raise


def _resolve_dashboard_dsn() -> str:
    postgres_dsn = os.getenv("POSTGRES_DSN", "").strip()
    if postgres_dsn:
        return postgres_dsn
    mysql_dsn = os.getenv("MYSQL_DSN", "").strip()
    if mysql_dsn:
        return mysql_dsn

    database_url = os.getenv("DATABASE_URL", "").strip()
    if database_url:
        if database_url.startswith("sqlite://") and not database_url.startswith("sqlite+aiosqlite://"):
            return database_url.replace("sqlite://", "sqlite+aiosqlite://")
        return database_url

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    db_path = os.path.join(data_dir, "reco_trading.db")
    return f"sqlite+aiosqlite:///{db_path}"


def _get_fallback_repository() -> Repository:
    global _fallback_repository
    if _fallback_repository is None:
        _fallback_repository = Repository(_resolve_dashboard_dsn())
        _run_async(_fallback_repository.setup())
    return _fallback_repository


def _get_repository_from_context() -> Repository | None:
    if _global_bot_instance is not None:
        repository = getattr(_global_bot_instance, "repository", None)
        if repository is not None:
            return repository
    try:
        return _get_fallback_repository()
    except Exception as exc:
        logger.error("Failed to initialize dashboard fallback repository: %s", exc)
        return None


def _fetch_trades(limit: int = 200, status_filter: str | None = None, symbol: str | None = None) -> tuple[list[Any], str]:
    """Fetch trades from database with optional status and symbol filters."""
    repository = _get_repository_from_context()
    if repository is None or not hasattr(repository, "get_trades"):
        return [], "snapshot"
    try:
        trades = _run_async(repository.get_trades(
            limit=limit,
            status_filter=status_filter,
            symbol=symbol
        ))
        if not isinstance(trades, list):
            return [], "snapshot"
        return trades, "database"
    except Exception as exc:
        logger.error("Error fetching trades from repository: %s", exc)
        return [], "snapshot"


def get_bot_snapshot() -> dict[str, Any]:
    """Get current snapshot from bot - enhanced with all fields."""
    global _global_bot_instance
    
    if _global_bot_instance is None and _bot_instance_getter is not None:
        try:
            _global_bot_instance = _bot_instance_getter()
        except Exception:
            pass
    
    if _global_bot_instance is None:
        snapshot = _get_default_snapshot()
        trades, source = _fetch_trades(limit=100)
        if source == "database" and trades:
            snapshot["trade_history"] = [
                {
                    "trade_id": t.id,
                    "pair": t.symbol,
                    "side": t.side,
                    "entry": float(t.entry_price) if t.entry_price else 0.0,
                    "exit": float(t.exit_price) if t.exit_price else 0.0,
                    "size": float(t.quantity) if t.quantity else 0.0,
                    "pnl": float(t.pnl) if t.pnl else 0.0,
                    "status": t.status,
                    "time": t.timestamp.strftime("%Y-%m-%d %H:%M") if t.timestamp else "-",
                }
                for t in trades
            ]
            snapshot["database_status"] = "CONNECTED"
        return snapshot
    
    try:
        # Thread-safe snapshot copy
        raw_snapshot = getattr(_global_bot_instance, 'snapshot', {})
        if callable(raw_snapshot):
            raw_snapshot = raw_snapshot()
        # Hacer copia shallow para evitar race conditions con el loop asyncio del bot
        snapshot = dict(raw_snapshot) if isinstance(raw_snapshot, dict) else {}
        
        # Add default values for missing fields
        snapshot = _enhance_snapshot(snapshot)
        snapshot.setdefault("status", "RUNNING")
        snapshot.setdefault("pair", "BTC/USDT")
        snapshot.setdefault("signal", "HOLD")
        
        return snapshot
    except Exception as e:
        logger.error(f"Error getting bot snapshot: {e}")
        return {
            "status": "ERROR",
            "pair": "BTC/USDT",
            "signal": "HOLD",
            "error": str(e),
        }


def _get_default_snapshot() -> dict[str, Any]:
    """Return default snapshot with all required fields."""
    return {
        "status": "WAITING",
        "pair": "BTC/USDT",
        "signal": "HOLD",
        "confidence": 0.0,
        "price": 0.0,
        "current_price": 0.0,
        "daily_pnl": 0.0,
        "session_pnl": 0.0,
        "equity": 0.0,
        "balance": 0.0,
        "total_equity": 0.0,
        "unrealized_pnl": 0.0,
        "win_rate": 0.0,
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "trades_today": 0,
        "has_open_position": False,
        "open_positions": [],
        "open_position_side": None,
        "open_position_entry": None,
        "open_position_qty": None,
        "open_position_sl": None,
        "open_position_tp": None,
        "position_size_pct": 0.0,
        "trend": "NEUTRAL",
        "momentum": "NEUTRAL",
        "volatility_regime": "NORMAL",
        "volatility_state": "NORMAL",
        "order_flow": "NEUTRAL",
        "volume_ratio": 1.0,
        "adx": 0.0,
        "atr": 0.0,
        "spread": 0.0,
        "rsi": 50.0,
        "macd_diff": 0.0,
        "ema_cross": "NEUTRAL",
        "cooldown": "READY",
        "exchange_status": "CONNECTED",
        "database_status": "CONNECTED",
        "capital_profile": "UNKNOWN",
        "operable_capital_usdt": 0.0,
        "capital_reserve_ratio": 0.0,
        "min_cash_buffer_usdt": 0.0,
        "risk_metrics": {},
        "runtime_settings": {},
        "trade_history": [],
        "logs": [],
        "timeframe": "5m",
        "loop_time_ms": 0,
        "api_latency_ms": 0,
        "uptime_seconds": 0,
        "btc_balance": 0.0,
        "btc_value": 0.0,
        "in_trade": 0.0,
        "last_trade": None,
        "ml_status": "Active",
        "ml_direction": None,
        "ml_confidence": 0.0,
        "ml_accuracy": 0.75,
        "market_regime": "UNKNOWN",
        "market_sentiment": "NEUTRAL",
        "confluence_score": 0.0,
        "change_24h": 0.0,
        "volume_24h": 0.0,
        "distance_to_support": 0.0,
        "distance_to_resistance": 0.0,
        "live_trade_risk_fraction": 0.01,
        "signals": {
            "trend": "NEUTRAL",
            "momentum": "NEUTRAL",
            "volume": "NEUTRAL",
            "volatility": "NEUTRAL",
            "structure": "NEUTRAL",
            "order_flow": "NEUTRAL",
        },
        "timeframe_analysis": {
            "5m": "NEUTRAL",
            "15m": "NEUTRAL",
            "1h": "NEUTRAL",
        },
        "signal_quality_score": 0.0,
        "raw_signal": "HOLD",
        "exit_intelligence_score": 0.0,
        "exit_intelligence_threshold": 0.0,
        "exit_intelligence_reason": None,
        "decision_reason": None,
        "decision_trace": {},
        "cooldown_complete": False,
        "capital_limit_usdt": 0.0,
        "circuit_breaker_trips": 0,
        "emergency_stop_active": False,
        "current_drawdown": 0.0,
        "max_drawdown": 0.10,
        "daily_loss_limit": 0.03,
        "min_confidence": 0.60,
        "capital_manager": {
            "capital_mode": "MEDIUM",
            "current_capital": 0.0,
            "initial_capital": 100.0,
            "peak_capital": 100.0,
            "current_drawdown_pct": 0.0,
            "win_streak": 0,
            "loss_streak": 0,
            "daily_trades": 0,
            "market_condition": "NORMAL",
            "effective_params": {
                "min_confidence": 0.62,
                "risk_per_trade": 0.012,
                "max_trades_per_day": 120,
            },
        },
        "smart_stop_stats": {
            "active_stops": 0,
            "trails_activated": 0,
            "break_evens_hit": 0,
            "profit_locks": 0,
            "last_stop_type": "-",
        },
        "smart_stop_status": "Ready",
        "ml_intelligence": {
            "metrics": {
                "accuracy": 0.75,
                "precision": 0.72,
                "recall": 0.70,
                "f1": 0.71,
            },
        },
        "active_blocks": [],
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "profit_factor": 0.0,
        "expectancy": 0.0,
        "candles_5m": [],
        "emergency_stop_active": False,
        "exchange_reconnections": 0,
        "dynamic_exit_enabled": True,
        "investment_mode": "Balanced",
        "optimized_risk_per_trade_fraction": 0.01,
        "optimization_reason": None,
        "api_latency_p95_ms": 0.0,
        "system": {},
        "decision_trace": {},
        "loop_time_ms": 0,
        "uptime_seconds": 0,
    }


def _enhance_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Add missing fields to snapshot with default values."""
    
    # Price and market data
    snapshot.setdefault("current_price", snapshot.get("price", 0.0))
    snapshot.setdefault("price", snapshot.get("price", 0.0))
    
    # PnL fields
    snapshot.setdefault("daily_pnl", snapshot.get("daily_pnl", 0.0))
    snapshot.setdefault("session_pnl", snapshot.get("session_pnl", 0.0))
    snapshot.setdefault("unrealized_pnl", snapshot.get("unrealized_pnl", 0.0))
    
    # Account fields
    snapshot.setdefault("equity", snapshot.get("equity", 0.0))
    snapshot.setdefault("balance", snapshot.get("balance", 0.0))
    snapshot.setdefault("total_equity", snapshot.get("total_equity", snapshot.get("equity", 0.0)))
    snapshot.setdefault("in_trade", snapshot.get("in_trade", 0.0))
    snapshot.setdefault("btc_balance", snapshot.get("btc_balance", 0.0))
    snapshot.setdefault("btc_value", snapshot.get("btc_value", 0.0))
    snapshot.setdefault("operable_capital_usdt", snapshot.get("operable_capital_usdt", 0.0))
    
    # Performance
    snapshot.setdefault("win_rate", snapshot.get("win_rate", 0.0))
    snapshot.setdefault("total_trades", snapshot.get("total_trades", snapshot.get("trades_today", 0)))
    snapshot.setdefault("wins", snapshot.get("wins", 0))
    snapshot.setdefault("losses", snapshot.get("losses", 0))
    snapshot.setdefault("trades_today", snapshot.get("trades_today", 0))
    snapshot.setdefault("avg_win", snapshot.get("avg_win", 0.0))
    snapshot.setdefault("avg_loss", snapshot.get("avg_loss", 0.0))
    snapshot.setdefault("profit_factor", snapshot.get("profit_factor", 0.0))
    snapshot.setdefault("expectancy", snapshot.get("expectancy", 0.0))
    snapshot.setdefault("last_trade", snapshot.get("last_trade", None))
    
    # Position
    snapshot.setdefault("has_open_position", snapshot.get("has_open_position", False))
    snapshot.setdefault("open_positions", snapshot.get("open_positions", []))
    snapshot.setdefault("open_position_side", snapshot.get("open_position_side", None))
    snapshot.setdefault("open_position_entry", snapshot.get("open_position_entry", 0.0))
    snapshot.setdefault("open_position_qty", snapshot.get("open_position_qty", 0.0))
    snapshot.setdefault("open_position_sl", snapshot.get("open_position_sl", 0.0))
    snapshot.setdefault("open_position_tp", snapshot.get("open_position_tp", 0.0))
    snapshot.setdefault("position_size_pct", snapshot.get("position_size_pct", 0.0))
    
    # Market indicators - separate fields
    snapshot.setdefault("trend", snapshot.get("trend", "NEUTRAL"))
    snapshot.setdefault("momentum", snapshot.get("momentum", "NEUTRAL"))
    snapshot.setdefault("volatility_regime", snapshot.get("volatility_regime", "NORMAL"))
    snapshot.setdefault("volatility_state", snapshot.get("volatility_state", snapshot.get("volatility_regime", "NORMAL")))
    snapshot.setdefault("order_flow", snapshot.get("order_flow", "NEUTRAL"))
    snapshot.setdefault("volume_ratio", snapshot.get("volume_ratio", 1.0))
    snapshot.setdefault("adx", snapshot.get("adx", 0.0))
    snapshot.setdefault("atr", snapshot.get("atr", 0.0))
    snapshot.setdefault("spread", snapshot.get("spread", 0.0))
    snapshot.setdefault("rsi", snapshot.get("rsi", 50.0))
    snapshot.setdefault("macd_diff", snapshot.get("macd_diff", 0.0))
    snapshot.setdefault("ema_cross", snapshot.get("ema_cross", "NEUTRAL"))
    snapshot.setdefault("change_24h", snapshot.get("change_24h", 0.0))
    snapshot.setdefault("volume_24h", snapshot.get("volume_24h", 0.0))
    
    # Signals - ensure proper structure
    signals = snapshot.get("signals", {})
    if not isinstance(signals, dict):
        signals = {}
    snapshot.setdefault("signals", {
        "trend": signals.get("trend", snapshot.get("trend", "NEUTRAL")),
        "momentum": signals.get("momentum", snapshot.get("momentum", "NEUTRAL")),
        "volume": signals.get("volume", "NEUTRAL"),
        "volatility": signals.get("volatility", "NEUTRAL"),
        "structure": signals.get("structure", "NEUTRAL"),
        "order_flow": signals.get("order_flow", snapshot.get("order_flow", "NEUTRAL")),
    })
    
    # Timeframe analysis
    tf_analysis = snapshot.get("timeframe_analysis", {})
    if not isinstance(tf_analysis, dict):
        tf_analysis = {}
    snapshot.setdefault("timeframe_analysis", {
        "5m": tf_analysis.get("5m", "NEUTRAL"),
        "15m": tf_analysis.get("15m", "NEUTRAL"),
        "1h": tf_analysis.get("1h", "NEUTRAL"),
    })
    
    # AI/ML
    snapshot.setdefault("ml_status", snapshot.get("ml_status", "Active"))
    snapshot.setdefault("ml_direction", snapshot.get("ml_direction", None))
    snapshot.setdefault("ml_confidence", snapshot.get("ml_confidence", 0.0))
    snapshot.setdefault("ml_accuracy", snapshot.get("ml_accuracy", 0.75))
    snapshot.setdefault("ml_predicted_move", snapshot.get("ml_predicted_move", 0.0))
    snapshot.setdefault("market_regime", snapshot.get("market_regime", "UNKNOWN"))
    snapshot.setdefault("market_sentiment", snapshot.get("market_sentiment", "NEUTRAL"))
    snapshot.setdefault("confluence_score", snapshot.get("confluence_score", 0.0))
    
    # Capital Manager
    cm = snapshot.get("capital_manager", {})
    if not isinstance(cm, dict):
        cm = {}
    snapshot.setdefault("capital_manager", {
        "capital_mode": cm.get("capital_mode", "MEDIUM"),
        "current_capital": cm.get("current_capital", snapshot.get("equity", 0.0)),
        "initial_capital": cm.get("initial_capital", 100.0),
        "peak_capital": cm.get("peak_capital", snapshot.get("equity", 0.0)),
        "current_drawdown_pct": cm.get("current_drawdown_pct", 0.0),
        "win_streak": cm.get("win_streak", 0),
        "loss_streak": cm.get("loss_streak", 0),
        "daily_trades": cm.get("daily_trades", snapshot.get("trades_today", 0)),
        "market_condition": cm.get("market_condition", "NORMAL"),
        "effective_params": cm.get("effective_params", {
            "min_confidence": 0.62,
            "risk_per_trade": 0.012,
            "max_trades_per_day": 120,
        }),
    })
    
    # Smart Stop Stats
    ss = snapshot.get("smart_stop_stats", {})
    if not isinstance(ss, dict):
        ss = {}
    snapshot.setdefault("smart_stop_stats", {
        "active_stops": ss.get("active_stops", 0),
        "trails_activated": ss.get("trails_activated", 0),
        "break_evens_hit": ss.get("break_evens_hit", 0),
        "profit_locks": ss.get("profit_locks", 0),
        "last_stop_type": ss.get("last_stop_type", "-"),
    })
    snapshot.setdefault("smart_stop_status", snapshot.get("smart_stop_status", "Ready"))
    
    # ML Intelligence
    ml_intel = snapshot.get("ml_intelligence", {})
    if not isinstance(ml_intel, dict):
        ml_intel = {}
    metrics = ml_intel.get("metrics", {})
    snapshot.setdefault("ml_intelligence", {
        "metrics": {
            "accuracy": metrics.get("accuracy", 0.75),
            "precision": metrics.get("precision", 0.72),
            "recall": metrics.get("recall", 0.70),
            "f1": metrics.get("f1", 0.71),
        },
    })
    
    # Risk metrics
    snapshot.setdefault("current_drawdown", snapshot.get("current_drawdown", 0.0))
    snapshot.setdefault("max_drawdown", snapshot.get("max_drawdown", 0.10))
    snapshot.setdefault("daily_loss_limit", snapshot.get("daily_loss_limit", 0.03))
    snapshot.setdefault("min_confidence", snapshot.get("min_confidence", 0.60))
    
    # System status
    snapshot.setdefault("status", snapshot.get("status", "RUNNING"))
    snapshot.setdefault("loop_time_ms", snapshot.get("loop_time_ms", 0))
    snapshot.setdefault("api_latency_ms", snapshot.get("api_latency_ms", 0))
    snapshot.setdefault("uptime_seconds", snapshot.get("uptime_seconds", 0))
    snapshot.setdefault("exchange_status", snapshot.get("exchange_status", "CONNECTED"))
    snapshot.setdefault("database_status", snapshot.get("database_status", "CONNECTED"))
    
    # Active blocks
    snapshot.setdefault("active_blocks", snapshot.get("active_blocks", []))
    
    # Exit intelligence
    snapshot.setdefault("exit_intelligence_score", snapshot.get("exit_intelligence_score", 0.0))
    snapshot.setdefault("exit_intelligence_threshold", snapshot.get("exit_intelligence_threshold", 0.0))
    snapshot.setdefault("exit_intelligence_reason", snapshot.get("exit_intelligence_reason", None))
    
    # Emergency
    snapshot.setdefault("circuit_breaker_trips", snapshot.get("circuit_breaker_trips", 0))
    snapshot.setdefault("emergency_stop_active", snapshot.get("emergency_stop_active", False))
    
    # Candles for chart
    snapshot.setdefault("candles_5m", snapshot.get("candles_5m", []))
    
    return snapshot
    snapshot.setdefault("stale_market_data_ratio", snapshot.get("stale_market_data_ratio", 0.0))
    
    # System
    system = snapshot.get("system", {}) or {}
    snapshot.setdefault("ui_lag_detected", system.get("ui_lag_detected", False))
    
    # Decision trace
    decision_trace = snapshot.get("decision_trace", {}) or {}
    snapshot.setdefault("factor_scores", decision_trace.get("factor_scores", {}))
    snapshot.setdefault("adaptive_size_multiplier", 
                       risk_metrics.get("adaptive_size_multiplier", 1.0))
    snapshot.setdefault("advanced_size_multiplier", 
                       risk_metrics.get("advanced_size_multiplier", 1.0))
    snapshot.setdefault("advanced_risk_reason", 
                       risk_metrics.get("advanced_risk_reason", "OK"))
    
    # Status fields
    snapshot.setdefault("cooldown", snapshot.get("cooldown", "READY"))
    snapshot.setdefault("exchange_status", snapshot.get("exchange_status", "CONNECTED"))
    snapshot.setdefault("database_status", snapshot.get("database_status", "CONNECTED"))
    snapshot.setdefault("status", snapshot.get("status", "RUNNING"))
    
    # Logs and trades
    snapshot.setdefault("logs", snapshot.get("logs", []))
    snapshot.setdefault("trade_history", snapshot.get("trade_history", []))
    
    return snapshot


async def broadcast_trade_execution(trade_data: dict[str, Any]) -> None:
    message = json.dumps(
        {
            "type": "trade_execution",
            "symbol": trade_data.get("symbol"),
            "side": trade_data.get("side"),
            "price": float(trade_data.get("price", 0.0)),
        }
    )
    stale_clients: list[Any] = []
    with _connected_clients_lock:
        clients = list(_connected_clients)
    for client in clients:
        try:
            client.send(message)
        except Exception as exc:
            logger.error("Error broadcasting to client: %s", exc)
            stale_clients.append(client)
    if stale_clients:
        with _connected_clients_lock:
            for client in stale_clients:
                if client in _connected_clients:
                    _connected_clients.remove(client)


def create_app() -> Flask:
    """Create and configure Flask app."""
    global _app, _sock

    # Resolve template and static folders relative to this file's location
    _web_site_dir = os.path.dirname(os.path.abspath(__file__))
    _template_dir = os.path.join(_web_site_dir, 'templates')
    _static_dir = os.path.join(_web_site_dir, 'static')

    app = Flask(__name__, template_folder=_template_dir, static_folder=_static_dir)
    app.config['JSON_SORT_KEYS'] = False
    if Sock is not None and _sock is None:
        _sock = Sock(app)

        @_sock.route("/ws")
        def websocket(ws):
            with _connected_clients_lock:
                _connected_clients.append(ws)
            try:
                while True:
                    if ws.receive() is None:
                        break
            finally:
                with _connected_clients_lock:
                    if ws in _connected_clients:
                        _connected_clients.remove(ws)

    # Global error handlers to prevent Internal Server Error pages
    @app.errorhandler(Exception)
    def _handle_exception(e):
        logger.exception("Unhandled exception in dashboard: %s", e)
        return jsonify({"success": False, "error": "Internal server error"}), 500

    @app.errorhandler(404)
    def _handle_404(e):
        return jsonify({"success": False, "error": "Not found"}), 404

    @app.errorhandler(500)
    def _handle_500(e):
        return jsonify({"success": False, "error": "Internal server error"}), 500

    @app.route("/")
    def index():
        try:
            return render_template("index.html")
        except Exception as e:
            logger.exception("Error rendering index: %s", e)
            return "<h1>Dashboard Error</h1><p>Could not load dashboard template.</p>", 500

    @app.route("/login", methods=["GET"])
    def serve_login():
        return render_template("login.html")

    @app.route("/api/auth/login", methods=["POST"])
    def api_auth_login():
        payload = request.get_json(silent=True) or {}
        username = str(payload.get("username") or payload.get("user") or "").strip()
        password = str(payload.get("password") or "").strip()
        configured_user = _dashboard_user()
        configured_password = _dashboard_password()
        if configured_user and configured_password:
            if hmac.compare_digest(username, configured_user) and hmac.compare_digest(password, configured_password):
                return jsonify({"success": True, "token": _create_jwt_token(username)})
        return jsonify({"success": False, "error": "Invalid credentials"}), 401
    
    @app.route('/api/snapshot')
    def api_snapshot():
        unauthorized = _require_dashboard_auth()
        if unauthorized:
            return unauthorized
        return jsonify({"success": True, "data": get_bot_snapshot()})
    

    @app.route('/api/stream')
    def api_stream():
        """SSE endpoint para actualizaciones en tiempo real con manejo de desconexión."""
        auth_err = _require_dashboard_auth()
        if auth_err:
            return auth_err

        def event_generator():
            import time
            consecutive_errors = 0
            while True:
                try:
                    data = get_bot_snapshot()
                    yield f"data: {json.dumps(data)}\n\n"
                    consecutive_errors = 0
                    time.sleep(0.3)
                except GeneratorExit:
                    logger.debug("SSE client disconnected cleanly")
                    break
                except Exception as e:
                    consecutive_errors += 1
                    logger.error("SSE error (#%d): %s", consecutive_errors, e)
                    if consecutive_errors >= 5:
                        logger.warning("SSE: demasiados errores consecutivos, cerrando stream")
                        break
                    try:
                        yield f"data: {json.dumps({'error': str(e), 'retry': 1000})}\n\n"
                    except Exception:
                        break
                    time.sleep(1)

        return Response(
            event_generator(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            },
        )

    @app.route("/api/status", methods=["GET"])
    @_require_auth
    def api_status():
        bot = _bot_instance_getter() if _bot_instance_getter else _global_bot_instance
        snapshot = get_bot_snapshot()
        return jsonify(
            {
                "symbol": str(getattr(bot, "symbol", snapshot.get("pair", "BTC/USDT"))),
                "status": snapshot.get("status", "UNKNOWN"),
                "balance": float(snapshot.get("balance", 0) or 0),
                "pnl": float(snapshot.get("daily_pnl", 0) or 0),
                "win_rate": float(snapshot.get("win_rate", 0) or 0),
                "open_positions": len(getattr(getattr(bot, "position_manager", None), "positions", []) or []),
            }
        )
    
    @app.route('/api/health')
    def api_health():
        snapshot = get_bot_snapshot()
        return jsonify({
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "exchange": snapshot.get("exchange_status", "UNKNOWN"),
            "database": snapshot.get("database_status", "UNKNOWN"),
            "api_latency_ms": snapshot.get("api_latency_p95_ms", 0),
        })
    
    @app.route('/api/control/<action>', methods=['POST'])
    def api_control(action):
        unauthorized = _require_dashboard_auth()
        if unauthorized:
            return unauthorized
        """Send control command to bot with instant reaction."""
        if _global_bot_instance is None:
            return jsonify({"success": False, "error": "Bot not connected"})
        
        try:
            bot = _global_bot_instance
            
            if action == 'pause':
                if hasattr(bot, 'snapshot'):
                    bot.snapshot["cooldown"] = "USER_PAUSED"
                    bot.snapshot["user_paused"] = True
                if hasattr(bot, 'manual_pause'):
                    bot.manual_pause = True
                logger.info("Bot paused via web dashboard")
                return jsonify({"success": True, "message": "Bot paused", "instant": True})
            
            elif action == 'resume':
                if hasattr(bot, 'snapshot'):
                    bot.snapshot["cooldown"] = None
                    bot.snapshot["user_paused"] = False
                    bot.snapshot["emergency_stop_active"] = False
                if hasattr(bot, 'manual_pause'):
                    bot.manual_pause = False
                if hasattr(bot, 'emergency_stop_active'):
                    bot.emergency_stop_active = False
                if hasattr(bot, 'trading_paused_by_drawdown'):
                    bot.trading_paused_by_drawdown = False
                if hasattr(bot, 'pause_trading_until'):
                    bot.pause_trading_until = None
                if hasattr(bot, 'exchange_failure_paused_until'):
                    bot.exchange_failure_paused_until = None
                logger.info("Bot resumed via web dashboard - all pause states cleared")
                return jsonify({"success": True, "message": "Bot resumed", "instant": True, "cleared_states": ["manual_pause", "emergency_stop", "drawdown_pause", "loss_pause", "exchange_pause"]})
            
            elif action == 'emergency':
                if hasattr(bot, 'snapshot'):
                    bot.snapshot["cooldown"] = "EMERGENCY_STOP"
                    bot.snapshot["emergency_stop_active"] = True
                if hasattr(bot, 'emergency_stop_active'):
                    bot.emergency_stop_active = True
                if hasattr(bot, 'manual_pause'):
                    bot.manual_pause = True
                logger.warning("Emergency stop activated via web dashboard")
                return jsonify({"success": True, "message": "Emergency stop activated", "instant": True})
            
            elif action in {'force_close', 'stop_trade'}:
                state_manager = getattr(bot, "state_manager", None)
                if state_manager and hasattr(state_manager, "request_force_close"):
                    state_manager.request_force_close()
                    logger.warning("Manual stop trade requested via web dashboard")
                    return jsonify({"success": True, "message": "Stop Trade request sent", "instant": True})
                elif hasattr(bot, "request_force_close"):
                    bot.request_force_close()
                    return jsonify({"success": True, "message": "Force close requested", "instant": True})
                else:
                    return jsonify({"success": False, "error": "Force close not available in this bot version"})
            
            elif action == 'clear_all_blocks':
                if hasattr(bot, 'snapshot'):
                    bot.snapshot["cooldown"] = None
                    bot.snapshot["user_paused"] = False
                    bot.snapshot["emergency_stop_active"] = False
                if hasattr(bot, 'manual_pause'):
                    bot.manual_pause = False
                if hasattr(bot, 'emergency_stop_active'):
                    bot.emergency_stop_active = False
                if hasattr(bot, 'trading_paused_by_drawdown'):
                    bot.trading_paused_by_drawdown = False
                if hasattr(bot, 'pause_trading_until'):
                    bot.pause_trading_until = None
                if hasattr(bot, 'exchange_failure_paused_until'):
                    bot.exchange_failure_paused_until = None
                if hasattr(bot, 'consecutive_losses'):
                    bot.consecutive_losses = 0
                logger.info("All blocks and pauses cleared via web dashboard")
                return jsonify({"success": True, "message": "All blocks cleared", "instant": True})
            
            else:
                return jsonify({"success": False, "error": "Unknown action"})
        except Exception as e:
            logger.error(f"Error sending control: {e}")
            return jsonify({"success": False, "error": str(e)})
    
    @app.route("/api/trades", methods=["GET"])
    @_require_auth
    def api_trades():
        try:
            status_filter = request.args.get("status", "").upper() or None
            symbol = request.args.get("symbol", "").upper() or None
            limit = min(int(request.args.get("limit", 200)), 500)
            offset = int(request.args.get("offset", 0))
            
            # Use status_filter for OPEN/CLOSED trades
            if status_filter and status_filter not in ("OPEN", "CLOSED"):
                status_filter = None
            
            trades, source = _fetch_trades(limit=limit, status_filter=status_filter, symbol=symbol)
            
            if source == "database" and trades:
                trade_rows = []
                for t in trades:
                    pnl = float(t.pnl or 0.0)
                    exit_price = float(t.exit_price or 0.0) if t.exit_price else float(t.entry_price or 0.0)
                    entry_price = float(t.entry_price or 0.0)
                    quantity = float(t.quantity or 0.0)
                    
                    # Calculate PnL percentage
                    if entry_price > 0 and quantity > 0:
                        pnl_pct = (pnl / (entry_price * quantity)) * 100
                    else:
                        pnl_pct = 0.0
                    
                    # Calculate duration
                    if t.close_timestamp and t.timestamp:
                        duration_seconds = (t.close_timestamp - t.timestamp).total_seconds()
                        duration_minutes = int(duration_seconds / 60)
                        if duration_minutes >= 60:
                            duration_str = f"{duration_minutes // 60}h {duration_minutes % 60}m"
                        else:
                            duration_str = f"{duration_minutes}m"
                    elif t.timestamp:
                        from datetime import datetime, timezone
                        duration_seconds = (datetime.now(timezone.utc) - t.timestamp).total_seconds()
                        duration_minutes = int(duration_seconds / 60)
                        if duration_minutes >= 60:
                            duration_str = f"{duration_minutes // 60}h {duration_minutes % 60}m"
                        else:
                            duration_str = f"{duration_minutes}m"
                    else:
                        duration_str = "-"
                    
                    # Calculate risk/reward ratio
                    if t.stop_loss and t.take_profit and entry_price > 0:
                        risk = abs(entry_price - float(t.stop_loss))
                        reward = abs(float(t.take_profit) - entry_price)
                        rr_ratio = f"{(reward / risk):.2f}R" if risk > 0 else "-"
                    else:
                        rr_ratio = "-"
                    
                    trade_rows.append({
                        "id": str(t.id),
                        "symbol": t.symbol or "UNKNOWN",
                        "side": t.side or "BUY",
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "current_price": exit_price,
                        "quantity": quantity,
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "status": str(t.status or "CLOSED").lower(),
                        "stop_loss": float(t.stop_loss or 0.0),
                        "take_profit": float(t.take_profit or 0.0),
                        "timestamp": (t.timestamp.isoformat() if t.timestamp else datetime.now().isoformat()),
                        "created_at": (t.timestamp.isoformat() if t.timestamp else datetime.now().isoformat()),
                        "close_timestamp": (t.close_timestamp.isoformat() if getattr(t, 'close_timestamp', None) and t.close_timestamp else None),
                        "duration": duration_str,
                        "rr_ratio": rr_ratio,
                        "order_id": t.order_id,
                    })
                
                # Sort by timestamp descending
                trade_rows.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                
                return jsonify({
                    "trades": trade_rows,
                    "source": "database",
                    "total": len(trade_rows),
                    "limit": limit,
                    "offset": offset
                })
            
            return jsonify({"trades": [], "source": "snapshot", "total": 0})
        except Exception as e:
            logger.error("Error getting trades: %s", e)
            import traceback
            traceback.print_exc()
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/trades/stats", methods=["GET"])
    @_require_auth
    def api_trade_stats():
        """Get trade statistics from database."""
        try:
            repository = _get_repository_from_context()
            if repository is None or not hasattr(repository, "get_trade_summary"):
                return jsonify({"success": False, "error": "Database not available"}), 503
            
            summary = _run_async(repository.get_trade_summary())
            
            return jsonify({
                "success": True,
                "stats": {
                    "total_trades": summary.get("total", 0),
                    "open_trades": summary.get("open", 0),
                    "closed_trades": summary.get("closed", 0),
                    "realized_pnl": float(summary.get("realized_pnl", 0)),
                    "win_rate": float(summary.get("win_rate", 0)),
                    "wins": summary.get("wins", 0),
                    "losses": summary.get("losses", 0),
                }
            })
        except Exception as e:
            logger.error("Error getting trade stats: %s", e)
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/db/status", methods=["GET"])
    @_require_auth
    def api_db_status():
        """Check database connection status."""
        try:
            repository = _get_repository_from_context()
            if repository is None:
                return jsonify({
                    "success": True,
                    "connected": False,
                    "message": "No repository available"
                })
            
            # Try to get a few trades to verify connection
            trades = _run_async(repository.get_trades(limit=1))
            return jsonify({
                "success": True,
                "connected": True,
                "message": f"Database connected, {len(trades)} trades visible"
            })
        except Exception as e:
            logger.error("Database status check error: %s", e)
            return jsonify({
                "success": True,
                "connected": False,
                "message": str(e)
            }), 500

    @app.route("/api/positions", methods=["GET"])
    @_require_auth
    def api_positions():
        """Get current open positions from bot and database."""
        try:
            positions = []
            
            # Get positions from bot instance
            bot = _bot_instance_getter() if _bot_instance_getter else _global_bot_instance
            if bot and hasattr(bot, 'position_manager'):
                pm = bot.position_manager
                if pm and hasattr(pm, 'positions') and pm.positions:
                    for pos in pm.positions:
                        positions.append({
                            "trade_id": pos.trade_id,
                            "symbol": getattr(bot, 'symbol', 'BTC/USDT'),
                            "side": pos.side,
                            "quantity": float(pos.quantity),
                            "entry_price": float(pos.entry_price),
                            "stop_loss": float(pos.stop_loss),
                            "take_profit": float(pos.take_profit),
                            "atr": float(pos.atr) if pos.atr else 0,
                            "unrealized_pnl": 0.0,  # Will be calculated from current price
                            "status": "OPEN",
                            "timestamp": pos.entry_timestamp_ms / 1000 if pos.entry_timestamp_ms else None,
                        })
            
            # Also get open positions from database
            repository = _get_repository_from_context()
            if repository:
                db_trades = _run_async(repository.get_trades(limit=50, status_filter="OPEN"))
                for t in db_trades:
                    # Check if already in positions from bot
                    if not any(p.get("trade_id") == t.id for p in positions):
                        positions.append({
                            "trade_id": t.id,
                            "symbol": t.symbol,
                            "side": t.side,
                            "quantity": float(t.quantity),
                            "entry_price": float(t.entry_price),
                            "stop_loss": float(t.stop_loss or 0),
                            "take_profit": float(t.take_profit or 0),
                            "atr": 0,
                            "unrealized_pnl": float(t.pnl or 0),
                            "status": "OPEN",
                            "timestamp": t.timestamp.isoformat() if t.timestamp else None,
                        })
            
            # Calculate unrealized PnL from snapshot price
            snapshot = get_bot_snapshot()
            current_price = float(snapshot.get("price", snapshot.get("current_price", 0)))
            
            for pos in positions:
                if current_price > 0 and pos["quantity"] > 0:
                    if pos["side"] == "BUY":
                        pos["unrealized_pnl"] = (current_price - pos["entry_price"]) * pos["quantity"]
                    else:
                        pos["unrealized_pnl"] = (pos["entry_price"] - current_price) * pos["quantity"]
            
            return jsonify({
                "success": True,
                "positions": positions,
                "count": len(positions)
            })
        except Exception as e:
            logger.error("Error getting positions: %s", e)
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/trades/<trade_id>/stop", methods=["POST"])
    @_require_auth
    def api_stop_trade(trade_id: str):
        bot = _bot_instance_getter() if _bot_instance_getter else _global_bot_instance
        if not bot:
            return jsonify({"success": False, "error": "Bot not available"}), 503
        try:
            state_manager = getattr(bot, "state_manager", None)
            if state_manager and hasattr(state_manager, "request_force_close"):
                state_manager.request_force_close()
                return jsonify({"success": True, "message": "Trade stop requested", "trade_id": trade_id})
            return jsonify({"success": False, "error": "Stop trade not available"}), 503
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
    
    @app.route('/api/all_trades')
    def api_all_trades():
        unauthorized = _require_dashboard_auth()
        if unauthorized:
            return unauthorized
        """Get all trades from database with pagination and filtering."""
        try:
            trades, source = _fetch_trades(limit=200)
            if source == "database":
                trade_list = []
                for t in trades:
                    entry_price = float(t.entry_price) if t.entry_price else 0
                    exit_price = float(t.exit_price) if t.exit_price else 0
                    quantity = float(t.quantity) if t.quantity else 0
                    pnl = float(t.pnl) if t.pnl else 0
                    pnl_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                    if t.side == "SELL":
                        pnl_pct = -pnl_pct
                    entry_time = t.timestamp.isoformat() if t.timestamp else None
                    exit_time = t.close_timestamp.isoformat() if t.close_timestamp else None
                    time_str = t.timestamp.strftime("%Y-%m-%d %H:%M") if t.timestamp else "-"
                    trade_list.append({
                        "trade_id": t.id,
                        "pair": t.symbol,
                        "side": t.side,
                        "entry": entry_price,
                        "exit": exit_price,
                        "size": quantity,
                        "stop_loss": float(t.stop_loss) if t.stop_loss else 0,
                        "take_profit": float(t.take_profit) if t.take_profit else 0,
                        "pnl": pnl,
                        "pnl_percent": pnl_pct,
                        "status": t.status,
                        "entry_time": entry_time,
                        "exit_time": exit_time,
                        "time": time_str,
                        "order_id": t.order_id,
                        "duration": _calc_duration(t.timestamp, t.close_timestamp) if t.close_timestamp else None,
                    })
                return jsonify({"trades": trade_list, "total": len(trade_list), "source": "database"})
            
            snapshot = get_bot_snapshot()
            trades = snapshot.get("trade_history", [])
            return jsonify({"trades": trades, "total": len(trades), "source": "snapshot"})
        except Exception as e:
            logger.error(f"Error getting all trades: {e}")
            return jsonify({"trades": [], "total": 0, "source": "error"})
    
    @app.route('/api/db_trades')
    def api_db_trades():
        unauthorized = _require_dashboard_auth()
        if unauthorized:
            return unauthorized
        """Get trades directly from database with pagination, filtering, and sorting."""
        try:
            repository = _get_repository_from_context()
            if not repository or not hasattr(repository, 'get_trades'):
                return jsonify({"trades": [], "total": 0})

            page = request.args.get('page', 1, type=int)
            per_page = request.args.get('per_page', 50, type=int)
            status_filter = request.args.get('status', '', type=str)
            pair_filter = request.args.get('pair', '', type=str)
            side_filter = request.args.get('side', '', type=str)
            
            per_page = min(per_page, 200)
            
            all_trades = _run_async(repository.get_trades(limit=1000))
            
            # Apply filters
            filtered = all_trades
            if status_filter:
                filtered = [t for t in filtered if t.status.upper() == status_filter.upper()]
            if pair_filter:
                filtered = [t for t in filtered if pair_filter.upper() in t.symbol.upper()]
            if side_filter:
                filtered = [t for t in filtered if t.side.upper() == side_filter.upper()]
            
            total = len(filtered)
            
            # Pagination
            start = (page - 1) * per_page
            end = start + per_page
            page_trades = filtered[start:end]
            
            trade_list = []
            for t in page_trades:
                entry_price = float(t.entry_price) if t.entry_price else 0
                exit_price = float(t.exit_price) if t.exit_price else 0
                quantity = float(t.quantity) if t.quantity else 0
                pnl = float(t.pnl) if t.pnl else 0
                pnl_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                
                if t.side == "SELL":
                    pnl_pct = -pnl_pct
                
                trade_list.append({
                    "trade_id": t.id,
                    "pair": t.symbol,
                    "side": t.side,
                    "entry": entry_price,
                    "exit": exit_price,
                    "size": quantity,
                    "stop_loss": float(t.stop_loss) if t.stop_loss else 0,
                    "take_profit": float(t.take_profit) if t.take_profit else 0,
                    "pnl": pnl,
                    "pnl_percent": pnl_pct,
                    "status": t.status,
                    "entry_time": t.timestamp.isoformat() if t.timestamp else None,
                    "exit_time": t.close_timestamp.isoformat() if t.close_timestamp else None,
                    "time": t.timestamp.strftime("%Y-%m-%d %H:%M") if t.timestamp else "-",
                    "order_id": t.order_id,
                })
            
            # Summary stats
            closed_trades = [t for t in filtered if t.status == "CLOSED" and t.pnl is not None]
            total_pnl = sum(float(t.pnl) for t in closed_trades) if closed_trades else 0
            winners = [t for t in closed_trades if float(t.pnl) > 0]
            losers = [t for t in closed_trades if float(t.pnl) <= 0]
            
            return jsonify({
                "trades": trade_list,
                "total": total,
                "page": page,
                "per_page": per_page,
                "total_pages": (total + per_page - 1) // per_page if per_page > 0 else 0,
                "summary": {
                    "total_trades": len(filtered),
                    "closed_trades": len(closed_trades),
                    "open_trades": len(filtered) - len(closed_trades),
                    "total_pnl": total_pnl,
                    "winners": len(winners),
                    "losers": len(losers),
                    "win_rate": len(winners) / len(closed_trades) * 100 if closed_trades else 0,
                }
            })
        except Exception as e:
            logger.error(f"Error getting db trades: {e}")
            return jsonify({"trades": [], "total": 0})
    
    @app.route('/api/trade_summary')
    def api_trade_summary():
        unauthorized = _require_dashboard_auth()
        if unauthorized:
            return unauthorized
        """Get trade summary statistics from database."""
        try:
            repository = _get_repository_from_context()
            if not repository or not hasattr(repository, 'get_trades'):
                return jsonify({})

            all_trades = _run_async(repository.get_trades(limit=1000))
            
            closed = [t for t in all_trades if t.status == "CLOSED" and t.pnl is not None]
            open_trades = [t for t in all_trades if t.status == "OPEN"]
            
            pnls = [float(t.pnl) for t in closed]
            winners = [p for p in pnls if p > 0]
            losers = [p for p in pnls if p <= 0]
            
            total_pnl = sum(pnls) if pnls else 0
            avg_win = sum(winners) / len(winners) if winners else 0
            avg_loss = sum(losers) / len(losers) if losers else 0
            best_trade = max(pnls) if pnls else 0
            worst_trade = min(pnls) if pnls else 0
            profit_factor = abs(sum(winners) / sum(losers)) if losers and sum(losers) != 0 else (abs(sum(winners)) if winners else 0)
            
            # Per-pair stats
            pair_stats = {}
            for t in closed:
                pair = t.symbol
                if pair not in pair_stats:
                    pair_stats[pair] = {"trades": 0, "pnl": 0, "wins": 0}
                pair_stats[pair]["trades"] += 1
                pair_stats[pair]["pnl"] += float(t.pnl)
                if float(t.pnl) > 0:
                    pair_stats[pair]["wins"] += 1
            
            return jsonify({
                "total_trades": len(all_trades),
                "closed_trades": len(closed),
                "open_trades": len(open_trades),
                "total_pnl": total_pnl,
                "win_rate": len(winners) / len(closed) * 100 if closed else 0,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "best_trade": best_trade,
                "worst_trade": worst_trade,
                "profit_factor": profit_factor,
                "avg_trade_duration": "-",
                "pair_stats": pair_stats,
            })
        except Exception as e:
            logger.error(f"Error getting trade summary: {e}")
            return jsonify({})
    
    @app.route('/api/analytics')
    def api_analytics():
        unauthorized = _require_dashboard_auth()
        if unauthorized:
            return unauthorized
        snapshot = get_bot_snapshot()
        analytics = snapshot.get("analytics", {})
        return jsonify(analytics)
    
    @app.route("/api/settings", methods=["GET"])
    @_require_auth
    def api_settings():
        bot = _bot_instance_getter() if _bot_instance_getter else _global_bot_instance
        if not bot:
            return jsonify({"success": False, "error": "Bot not available"}), 503
        return jsonify(
            {
                "symbol": str(getattr(bot, "symbol", "BTC/USDT")),
                "max_position_size": float(getattr(getattr(bot, "settings", None), "max_trade_balance_fraction", 0.0) or 0.0),
                "risk_per_trade": float(getattr(getattr(bot, "settings", None), "risk_per_trade_fraction", 0.0) or 0.0),
                "enable_short": not bool(getattr(getattr(bot, "settings", None), "spot_only_mode", True)),
            }
        )

    @app.route("/api/settings/pair", methods=["POST"])
    @_require_auth
    def api_update_pair():
        bot = _bot_instance_getter() if _bot_instance_getter else _global_bot_instance
        if not bot:
            return jsonify({"success": False, "error": "Bot not available"}), 503
        data = request.get_json(silent=True) or {}
        new_symbol = str(data.get("symbol", "")).strip()
        if not new_symbol:
            return jsonify({"success": False, "error": "Symbol required"}), 400
        try:
            bot.symbol = new_symbol
            setattr(bot, "_pending_pair_switch", new_symbol)
            return jsonify({"success": True, "message": f"Pair updated to {new_symbol}"})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
    
    @app.route('/api/autonomous')
    def api_autonomous():
        unauthorized = _require_dashboard_auth()
        if unauthorized:
            return unauthorized
        """Get autonomous brain status including optimizers and regime detector."""
        global _global_bot_instance, _bot_instance_getter
        bot = _global_bot_instance or (_bot_instance_getter() if _bot_instance_getter else None)
        if not bot or not hasattr(bot, 'autonomous_brain'):
            return jsonify({"error": "Autonomous brain not available"})
        try:
            return jsonify(bot.autonomous_brain.get_status())
        except Exception as e:
            logger.error(f"Error getting autonomous status: {e}")
            return jsonify({"error": str(e)})
    
    @app.route('/api/multipair')
    def api_multipair():
        unauthorized = _require_dashboard_auth()
        if unauthorized:
            return unauthorized
        """Get multi-pair manager status."""
        global _global_bot_instance, _bot_instance_getter
        bot = _global_bot_instance or (_bot_instance_getter() if _bot_instance_getter else None)
        if not bot or not hasattr(bot, 'multi_pair_manager'):
            return jsonify({"error": "Multi-pair manager not available"})
        try:
            mpm = bot.multi_pair_manager
            pairs_data = []
            default_pairs = getattr(mpm, 'default_pairs', ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'])
            pairs_metrics = getattr(mpm, 'pairs_metrics', {})
            
            for symbol in default_pairs:
                if symbol in pairs_metrics:
                    m = pairs_metrics[symbol]
                    pairs_data.append({
                        "symbol": symbol,
                        "opportunity_score": float(getattr(m, 'opportunity_score', 0)),
                        "volatility": float(getattr(m, 'volatility', 0)),
                        "momentum_score": float(getattr(m, 'momentum_score', 0)),
                        "volume_score": float(getattr(m, 'volume_score', 0)),
                        "trend_direction": getattr(m, 'trend_direction', 'NEUTRAL'),
                        "rsi": float(getattr(m, 'rsi', 50)),
                    })
            return jsonify({
                "active_pair": getattr(mpm, 'active_pair', 'BTC/USDT'),
                "scan_interval": getattr(mpm, 'scan_interval_seconds', 60),
                "pairs": pairs_data,
            })
        except Exception as e:
            logger.error(f"Error getting multipair status: {e}")
            return jsonify({"error": str(e)})
    
    @app.route('/api/market_data')
    def api_market_data():
        unauthorized = _require_dashboard_auth()
        if unauthorized:
            return unauthorized
        """Get detailed market data."""
        snapshot = get_bot_snapshot()
        return jsonify({
            "price": snapshot.get("current_price", 0),
            "rsi": snapshot.get("rsi", 50),
            "adx": snapshot.get("adx", 0),
            "spread": snapshot.get("spread", 0),
            "volume_24h": snapshot.get("volume_24h", 0),
            "change_24h": snapshot.get("change_24h", 0),
            "trend": snapshot.get("trend", "NEUTRAL"),
            "volatility_regime": snapshot.get("volatility_regime", "NORMAL"),
            "order_flow": snapshot.get("order_flow", "NEUTRAL"),
            "distance_to_support": snapshot.get("distance_to_support", 0),
            "distance_to_resistance": snapshot.get("distance_to_resistance", 0),
        })
    
    @app.route('/api/risk_data')
    def api_risk_data():
        unauthorized = _require_dashboard_auth()
        if unauthorized:
            return unauthorized
        """Get detailed risk data."""
        snapshot = get_bot_snapshot()
        risk_metrics = snapshot.get("risk_metrics", {}) or {}
        
        return jsonify({
            "capital_profile": snapshot.get("capital_profile", "UNKNOWN"),
            "operable_capital_usdt": snapshot.get("operable_capital_usdt", 0),
            "capital_reserve_ratio": snapshot.get("capital_reserve_ratio", 0),
            "min_cash_buffer_usdt": snapshot.get("min_cash_buffer_usdt", 0),
            "capital_limit_usdt": snapshot.get("capital_limit_usdt", 0),
            "live_trade_risk_fraction": snapshot.get("live_trade_risk_fraction", 0.01),
            "optimized_risk_per_trade_fraction": snapshot.get("optimized_risk_per_trade_fraction", 0.01),
            "optimization_reason": snapshot.get("optimization_reason", None),
            "investment_mode": snapshot.get("investment_mode", "Balanced"),
            "adaptive_size_multiplier": risk_metrics.get("adaptive_size_multiplier", 1.0),
            "advanced_size_multiplier": risk_metrics.get("advanced_size_multiplier", 1.0),
            "advanced_risk_reason": risk_metrics.get("advanced_risk_reason", "OK"),
            "current_exposure": risk_metrics.get("current_exposure", 0),
            "max_drawdown": risk_metrics.get("max_drawdown", 0),
        })
    
    @app.route('/api/health_detailed')
    def api_health_detailed():
        unauthorized = _require_dashboard_auth()
        if unauthorized:
            return unauthorized
        """Get detailed health metrics."""
        snapshot = get_bot_snapshot()
        
        return jsonify({
            "exchange_status": snapshot.get("exchange_status", "UNKNOWN"),
            "database_status": snapshot.get("database_status", "UNKNOWN"),
            "api_latency_p95_ms": snapshot.get("api_latency_p95_ms", 0),
            "api_latency_ms": snapshot.get("api_latency_ms", 0),
            "ui_render_ms": snapshot.get("ui_render_ms", 0),
            "ui_staleness_ms": snapshot.get("ui_staleness_ms", 0),
            "stale_market_data_ratio": snapshot.get("stale_market_data_ratio", 0),
            "ui_lag_detected": snapshot.get("ui_lag_detected", False),
            "circuit_breaker_trips": snapshot.get("circuit_breaker_trips", 0),
            "emergency_stop_active": snapshot.get("emergency_stop_active", False),
            "exchange_reconnections": snapshot.get("exchange_reconnections", 0),
            "dynamic_exit_enabled": snapshot.get("dynamic_exit_enabled", True),
        })
    

    @app.route('/api/capital_profile')
    def api_capital_profile():
        """Expone el perfil de capital activo y estado de filtros adaptativos."""
        unauthorized = _require_dashboard_auth()
        if unauthorized:
            return unauthorized
        snapshot = get_bot_snapshot()
        return jsonify({
            "capital_profile":          snapshot.get("capital_profile", "UNKNOWN"),
            "operable_capital_usdt":    snapshot.get("operable_capital_usdt", 0.0),
            "capital_reserve_ratio":    snapshot.get("capital_reserve_ratio", 0.0),
            "min_cash_buffer_usdt":     snapshot.get("min_cash_buffer_usdt", 0.0),
            "capital_limit_usdt":       snapshot.get("capital_limit_usdt", 0.0),
            "filter_relaxation_active": snapshot.get("filter_relaxation_active", False),
            "filter_auto_adjustments":  snapshot.get("filter_auto_adjustments", 0),
            "trades_target_daily":      snapshot.get("trades_target_daily", 0),
            "trades_today":             snapshot.get("trades_today", 0),
            "autonomous_filters":       snapshot.get("autonomous_filters", {}),
        })

    @app.route("/api/risk", methods=["GET"])
    @_require_auth
    def api_risk():
        bot = _bot_instance_getter() if _bot_instance_getter else _global_bot_instance
        snapshot = get_bot_snapshot()
        risk_metrics = snapshot.get("risk_metrics", {}) or {}
        
        return jsonify({
            "daily_limit": float(getattr(getattr(bot, "settings", None), "daily_loss_limit_fraction", 0.03) or 0.03),
            "daily_used": float(snapshot.get("daily_pnl", 0) / max(snapshot.get("balance", 1), 1) if snapshot.get("balance", 0) > 0 else 0),
            "max_positions": int(getattr(getattr(bot, "settings", None), "max_concurrent_trades", 1) or 1),
            "current_positions": len(snapshot.get("open_positions", [])),
            "exposure": float(risk_metrics.get("current_exposure", 0)),
            "base_size": float(risk_metrics.get("base_risk_fraction", 0.01)),
            "multiplier": float(risk_metrics.get("adaptive_size_multiplier", 1.0)),
            "effective_size": float(risk_metrics.get("effective_risk_fraction", 0.01)),
            "active_blocks": _get_active_blocks(snapshot),
        })

    @app.route("/api/ml", methods=["GET"])
    @_require_auth
    def api_ml():
        snapshot = get_bot_snapshot()
        ml_intel = snapshot.get("ml_intelligence", {}) or {}
        
        return jsonify({
            "ensemble_status": "Active" if snapshot.get("ml_direction") else "Idle",
            "last_prediction": snapshot.get("ml_direction", "-"),
            "last_confidence": float(snapshot.get("ml_confidence", 0)),
            "accuracy_7d": float(ml_intel.get("metrics", {}).get("accuracy", 0.75) if isinstance(ml_intel.get("metrics"), dict) else 0.75),
            "precision": float(ml_intel.get("metrics", {}).get("precision", 0.72) if isinstance(ml_intel.get("metrics"), dict) else 0.72),
            "recall": float(ml_intel.get("metrics", {}).get("recall", 0.70) if isinstance(ml_intel.get("metrics"), dict) else 0.70),
            "f1_score": float(ml_intel.get("metrics", {}).get("f1", 0.71) if isinstance(ml_intel.get("metrics"), dict) else 0.71),
            "training_samples": 1000,
            "predictions": [],
        })

    @app.route("/api/candles", methods=["GET"])
    @_require_auth
    def api_candles():
        global _global_bot_instance
        try:
            limit = request.args.get("limit", 200, type=int)
            timeframe = request.args.get("timeframe", "5m")
            limit = min(max(limit, 10), 500)
            
            snapshot = get_bot_snapshot()
            candles_raw = snapshot.get("candles_5m", [])
            
            if candles_raw and isinstance(candles_raw, list) and len(candles_raw) > 0:
                candles = []
                for c in candles_raw[-limit:]:
                    try:
                        if isinstance(c, dict):
                            ts = c.get("timestamp") or c.get("time") or c.get("t")
                            if ts is None:
                                continue
                            if isinstance(ts, datetime):
                                ts_int = int(ts.timestamp())
                            elif isinstance(ts, (int, float)):
                                ts_int = int(ts)
                            else:
                                try:
                                    ts_int = int(float(ts))
                                except (ValueError, TypeError):
                                    continue
                            
                            o = float(c.get("open", 0))
                            h = float(c.get("high", 0))
                            l = float(c.get("low", 0))
                            cl = float(c.get("close", 0))
                            
                            if o > 0 and h > 0 and l > 0 and cl > 0:
                                candles.append({
                                    "time": ts_int,
                                    "open": o,
                                    "high": h,
                                    "low": l,
                                    "close": cl,
                                })
                    except (ValueError, TypeError, KeyError):
                        continue
                
                return jsonify({"candles": candles, "count": len(candles)})
            
            return jsonify({"candles": [], "count": 0})
        except Exception as e:
            logger.error(f"Error getting candles: {e}")
            return jsonify({"candles": [], "count": 0})

    @app.route("/api/logs", methods=["GET"])
    @_require_auth
    def api_logs():
        limit = request.args.get("limit", 100, type=int)
        limit = min(limit, 500)
        
        snapshot = get_bot_snapshot()
        logs_raw = snapshot.get("logs", [])
        
        logs = []
        for log in logs_raw[-limit:]:
            if isinstance(log, dict):
                logs.append({
                    "timestamp": log.get("timestamp", ""),
                    "level": log.get("level", "INFO"),
                    "message": log.get("message", str(log)),
                })
            else:
                logs.append({
                    "timestamp": "",
                    "level": "INFO",
                    "message": str(log),
                })
        
        return jsonify({"logs": logs})

    @app.route("/api/settings", methods=["POST"])
    @_require_auth
    def api_update_settings():
        bot = _bot_instance_getter() if _bot_instance_getter else _global_bot_instance
        if not bot:
            return jsonify({"success": False, "error": "Bot not available"}), 503
        
        data = request.get_json(silent=True) or {}
        
        try:
            settings = getattr(bot, "settings", None)
            if settings:
                if "risk_per_trade" in data and hasattr(settings, "risk_per_trade_fraction"):
                    setattr(settings, "risk_per_trade_fraction", float(data["risk_per_trade"]))
                if "max_daily_loss" in data and hasattr(settings, "daily_loss_limit_fraction"):
                    setattr(settings, "daily_loss_limit_fraction", float(data["max_daily_loss"]))
                if "max_drawdown" in data and hasattr(settings, "max_drawdown_fraction"):
                    setattr(settings, "max_drawdown_fraction", float(data["max_drawdown"]))
                if "max_trades_day" in data and hasattr(settings, "max_trades_per_day"):
                    setattr(settings, "max_trades_per_day", int(data["max_trades_day"]))
                if "min_confidence" in data and hasattr(settings, "min_signal_confidence"):
                    setattr(settings, "min_signal_confidence", float(data["min_confidence"]))
                if "adx_threshold" in data and hasattr(settings, "adx_min_threshold"):
                    setattr(settings, "adx_min_threshold", float(data["adx_threshold"]))
                if "stop_atr_multiplier" in data:
                    bot.snapshot["stop_atr_multiplier"] = float(data["stop_atr_multiplier"])
                if "trailing_activation_r" in data:
                    bot.snapshot["trailing_activation_r"] = float(data["trailing_activation_r"])
                if "breakeven_trigger_r" in data:
                    bot.snapshot["breakeven_trigger_r"] = float(data["breakeven_trigger_r"])
                if "profit_lock_trigger_r" in data:
                    bot.snapshot["profit_lock_trigger_r"] = float(data["profit_lock_trigger_r"])
                if "ml_enabled" in data:
                    bot.snapshot["ml_enabled"] = bool(data["ml_enabled"])
                if "trading_mode" in data:
                    bot.snapshot["trading_mode"] = data["trading_mode"]
                if "primary_timeframe" in data and hasattr(settings, "primary_timeframe"):
                    setattr(settings, "timeframe", data["primary_timeframe"])
                if "loop_sleep_seconds" in data and hasattr(settings, "loop_sleep_seconds"):
                    setattr(settings, "loop_sleep_seconds", int(data["loop_sleep_seconds"]))
            
            return jsonify({"success": True})
        except Exception as e:
            logger.error(f"Error updating settings: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/control/start", methods=["POST"])
    def api_control_start():
        unauthorized = _require_dashboard_auth()
        if unauthorized:
            return unauthorized
        
        global _global_bot_instance
        if _global_bot_instance is None:
            return jsonify({"success": False, "error": "Bot not connected"})
        
        try:
            if hasattr(_global_bot_instance, 'snapshot'):
                _global_bot_instance.snapshot["cooldown"] = None
                _global_bot_instance.snapshot["user_paused"] = False
                _global_bot_instance.snapshot["emergency_stop_active"] = False
            if hasattr(_global_bot_instance, 'emergency_stop_active'):
                _global_bot_instance.emergency_stop_active = False
            if hasattr(_global_bot_instance, 'manual_pause'):
                _global_bot_instance.manual_pause = False
            
            logger.info("Bot started via web dashboard")
            return jsonify({"success": True, "message": "Bot started"})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/control/pause", methods=["POST"])
    def api_control_pause():
        unauthorized = _require_dashboard_auth()
        if unauthorized:
            return unauthorized
        
        global _global_bot_instance
        if _global_bot_instance is None:
            return jsonify({"success": False, "error": "Bot not connected"})
        
        try:
            if hasattr(_global_bot_instance, 'manual_pause'):
                _global_bot_instance.manual_pause = True
            if hasattr(_global_bot_instance, 'snapshot'):
                _global_bot_instance.snapshot["cooldown"] = "USER_PAUSED"
                _global_bot_instance.snapshot["user_paused"] = True
            
            logger.info("Bot paused via web dashboard")
            return jsonify({"success": True, "message": "Bot paused"})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/control/stop", methods=["POST"])
    def api_control_stop():
        unauthorized = _require_dashboard_auth()
        if unauthorized:
            return unauthorized
        
        global _global_bot_instance
        if _global_bot_instance is None:
            return jsonify({"success": False, "error": "Bot not connected"})
        
        try:
            if hasattr(_global_bot_instance, 'snapshot'):
                _global_bot_instance.snapshot["cooldown"] = "EMERGENCY_STOP"
                _global_bot_instance.snapshot["emergency_stop_active"] = True
            if hasattr(_global_bot_instance, 'emergency_stop_active'):
                _global_bot_instance.emergency_stop_active = True
            
            logger.warning("Bot emergency stop via web dashboard")
            return jsonify({"success": True, "message": "Emergency stop activated"})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/support/diagnostics", methods=["GET"])
    @_require_auth
    def api_support_diagnostics():
        """Run comprehensive diagnostics and return results."""
        try:
            from reco_trading.support.support_manager import SupportManager
            bot = _bot_instance_getter() if _bot_instance_getter else _global_bot_instance
            support = SupportManager(bot)
            
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    diagnostics = list(support._diagnostics_cache.values())
                    if not diagnostics:
                        diagnostics = [{"name": "pending", "status": "ok", "message": "Run diagnostics from background"}]
                else:
                    diagnostics = loop.run_until_complete(support.run_diagnostics(force=True))
            except Exception:
                diagnostics = list(support._diagnostics_cache.values())
            
            return jsonify({
                "success": True,
                "diagnostics": [
                    {"name": d.name, "status": d.status, "message": d.message, "details": d.details}
                    for d in diagnostics
                ],
                "system_info": {
                    "os": support.get_system_info().os_name,
                    "python": support.get_system_info().python_version,
                    "cpu_count": support.get_system_info().cpu_count,
                    "memory_gb": support.get_system_info().memory_total_gb,
                },
            })
        except Exception as e:
            logger.error(f"Error running diagnostics: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/support/help/<issue>", methods=["GET"])
    def api_support_help(issue: str):
        """Get help documentation for a specific issue."""
        try:
            from reco_trading.support.support_manager import SupportManager
            support = SupportManager()
            help_doc = support.get_help_for_issue(issue)
            return jsonify({
                "success": True,
                "issue": issue,
                "help": help_doc,
            })
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/support/ticket", methods=["POST"])
    @_require_auth
    def api_support_create_ticket():
        """Create a support ticket with system info and diagnostics."""
        try:
            from reco_trading.support.support_manager import SupportManager
            bot = _bot_instance_getter() if _bot_instance_getter else _global_bot_instance
            support = SupportManager(bot)
            
            data = request.get_json(silent=True) or {}
            title = str(data.get("title", "Support Request"))
            description = str(data.get("description", ""))
            priority = str(data.get("priority", "medium"))
            category = str(data.get("category", "bug"))
            include_logs = bool(data.get("include_logs", True))
            include_diagnostics = bool(data.get("include_diagnostics", True))
            
            ticket = support.create_support_ticket(
                title=title,
                description=description,
                priority=priority,
                category=category,
                include_logs=include_logs,
                include_diagnostics=include_diagnostics,
            )
            
            exported = support.export_ticket(ticket)
            
            return jsonify({
                "success": True,
                "ticket_id": ticket.ticket_id,
                "status": ticket.status,
                "export": exported,
            })
        except Exception as e:
            logger.error(f"Error creating support ticket: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/pause_status", methods=["GET"])
    @_require_auth
    def api_pause_status():
        """Get detailed pause status - NEVER pauses, only returns current state."""
        try:
            snapshot = get_bot_snapshot()
            pause_states = snapshot.get("pause_states", {})
            auto_pause_disabled = snapshot.get("auto_pause_disabled", True)
            
            any_paused = any([
                pause_states.get("manual_pause", False),
                pause_states.get("emergency_stop", False),
                pause_states.get("user_paused", False),
                pause_states.get("drawdown_pause", False),
                pause_states.get("loss_pause", False),
                pause_states.get("exchange_pause", False),
            ])
            
            cooldown = snapshot.get("cooldown")
            
            return jsonify({
                "success": True,
                "is_paused": any_paused or bool(cooldown),
                "auto_pause_disabled": auto_pause_disabled,
                "pause_states": pause_states,
                "cooldown": cooldown,
                "can_trade": not any_paused and not cooldown,
                "message": "Bot is running" if not any_paused and not cooldown else f"Bot paused: {cooldown or 'unknown'}",
            })
        except Exception as e:
            logger.error(f"Error getting pause status: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    def _get_active_blocks(snapshot: dict) -> list:
        blocks = []
        if snapshot.get("emergency_stop_active"):
            blocks.append("EMERGENCY_STOP")
        if snapshot.get("user_paused"):
            blocks.append("USER_PAUSED")
        if snapshot.get("exchange_failure_paused_until"):
            blocks.append("EXCHANGE_PAUSED")
        if snapshot.get("circuit_breaker_trips", 0) > 0:
            blocks.append(f"CIRCUIT_BREAKER ({snapshot.get('circuit_breaker_trips')})")
        daily_pnl = snapshot.get("daily_pnl", 0)
        balance = snapshot.get("balance", 1)
        if balance > 0 and daily_pnl < -(balance * 0.03):
            blocks.append("DAILY_LOSS_LIMIT")
        return blocks

    _app = app
    return app


def run_server(host: str = '0.0.0.0', port: int = 9000) -> None:
    """Run the Flask dashboard server."""
    app = create_app()
    werkzeug_logger = logging.getLogger("werkzeug")
    werkzeug_logger.disabled = True
    app.logger.disabled = True

    def _kill_process_on_port(target_port: int) -> bool:
        """Kill any process listening on the target port."""
        try:
            # Method 1: lsof
            import subprocess
            result = subprocess.run(
                ["lsof", "-ti", f":{target_port}"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split("\n")
                for pid in pids:
                    pid = pid.strip()
                    if pid:
                        try:
                            os.kill(int(pid), 9)
                            logger.info(f"Killed process {pid} on port {target_port}")
                        except (ProcessLookupError, PermissionError, ValueError):
                            pass
                time.sleep(0.5)
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        try:
            # Method 2: fuser
            import subprocess
            result = subprocess.run(
                ["fuser", "-k", f"{target_port}/tcp"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                logger.info(f"Killed process on port {target_port} via fuser")
                time.sleep(0.5)
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        try:
            # Method 3: ss + kill
            import subprocess
            result = subprocess.run(
                ["ss", "-tlnp", f"sport = :{target_port}"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and str(target_port) in result.stdout:
                import re
                pids_found = re.findall(r'pid=(\d+)', result.stdout)
                for pid in pids_found:
                    try:
                        os.kill(int(pid), 9)
                        logger.info(f"Killed process {pid} on port {target_port} via ss")
                    except (ProcessLookupError, PermissionError, ValueError):
                        pass
                time.sleep(0.5)
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return False

    def find_available_port(start_port: int) -> int:
        """Kill process on start_port if in use, then return start_port. Fallback to next available."""
        # First try to free the requested port
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((host, start_port))
            sock.close()
            logger.info(f"Port {start_port} is available")
            return start_port
        except OSError:
            # Port is in use - try to kill the process
            logger.warning(f"Port {start_port} is in use, attempting to free it...")
            _kill_process_on_port(start_port)
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind((host, start_port))
                sock.close()
                logger.info(f"Successfully freed port {start_port}")
                return start_port
            except OSError:
                pass

        # Fallback: find next available port
        port = start_port + 1
        for _ in range(100):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind((host, port))
                sock.close()
                logger.info(f"Found available port: {port}")
                return port
            except OSError:
                port += 1
        raise RuntimeError(f"Could not find available port near {start_port}")

    actual_port = find_available_port(port)

    if actual_port != port:
        logger.warning(f"Port {port} was in use, using port {actual_port} instead")

    # Print clear startup message
    import sys
    print("", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print("  WEB DASHBOARD STARTED", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Local:   http://127.0.0.1:{actual_port}", file=sys.stderr)
    print(f"  Network: http://0.0.0.0:{actual_port}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print("", file=sys.stderr)
    logger.info(f"Web dashboard started on http://{host}:{actual_port}")

    app.run(host=host, port=actual_port, debug=False, use_reloader=False, threaded=True)


def run_in_thread(host: str = '0.0.0.0', port: int = 9000) -> threading.Thread:
    """Run the Flask server in a background thread."""
    thread = threading.Thread(target=run_server, args=(host, port), daemon=True, name="web-dashboard")
    thread.start()
    return thread


if __name__ == '__main__':
    run_server()
