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


def _fetch_trades(limit: int = 200) -> tuple[list[Any], str]:
    repository = _get_repository_from_context()
    if repository is None or not hasattr(repository, "get_trades"):
        return [], "snapshot"
    try:
        trades = _run_async(repository.get_trades(limit=limit))
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
        "trades_today": 0,
        "has_open_position": False,
        "open_position_side": None,
        "open_position_entry": None,
        "open_position_qty": None,
        "open_position_sl": None,
        "open_position_tp": None,
        "trend": "NEUTRAL",
        "volatility_regime": "NORMAL",
        "order_flow": "NEUTRAL",
        "adx": 0.0,
        "spread": 0.0,
        "rsi": 50.0,
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
        "timeframe": "5m / 15m",
        # Extended fields
        "btc_balance": 0.0,
        "btc_value": 0.0,
        "last_trade": None,
        "ml_direction": None,
        "ml_confidence": 0.0,
        "market_regime": "UNKNOWN",
        "market_sentiment": "NEUTRAL",
        "confluence_score": 0.0,
        "change_24h": 0.0,
        "volume_24h": 0.0,
        "distance_to_support": 0.0,
        "distance_to_resistance": 0.0,
        "live_trade_risk_fraction": 0.01,
        "signals": {},
        "signal_quality_score": 0.0,
        "raw_signal": "HOLD",
        "exit_intelligence_score": 0.0,
        "exit_intelligence_threshold": 0.0,
        "exit_intelligence_reason": None,
        "decision_reason": None,
        "cooldown_complete": False,
        "capital_limit_usdt": 0.0,
        "circuit_breaker_trips": 0,
        "emergency_stop_active": False,
        "exchange_reconnections": 0,
        "dynamic_exit_enabled": True,
        "investment_mode": "Balanced",
        "optimized_risk_per_trade_fraction": 0.01,
        "optimization_reason": None,
        "api_latency_p95_ms": 0.0,
        "system": {},
        "decision_trace": {},
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
    snapshot.setdefault("total_equity", snapshot.get("total_equity", 0.0))
    snapshot.setdefault("btc_balance", snapshot.get("btc_balance", 0.0))
    snapshot.setdefault("btc_value", snapshot.get("btc_value", 0.0))
    snapshot.setdefault("operable_capital_usdt", snapshot.get("operable_capital_usdt", 0.0))
    
    # Performance
    snapshot.setdefault("win_rate", snapshot.get("win_rate", 0.0))
    snapshot.setdefault("trades_today", snapshot.get("trades_today", 0))
    snapshot.setdefault("last_trade", snapshot.get("last_trade", None))
    
    # Position
    snapshot.setdefault("has_open_position", snapshot.get("has_open_position", False))
    snapshot.setdefault("open_position_side", snapshot.get("open_position_side", None))
    snapshot.setdefault("open_position_entry", snapshot.get("open_position_entry", 0.0))
    snapshot.setdefault("open_position_qty", snapshot.get("open_position_qty", 0.0))
    snapshot.setdefault("open_position_sl", snapshot.get("open_position_sl", 0.0))
    snapshot.setdefault("open_position_tp", snapshot.get("open_position_tp", 0.0))
    
    # Market indicators
    snapshot.setdefault("trend", snapshot.get("trend", "NEUTRAL"))
    snapshot.setdefault("adx", snapshot.get("adx", 0.0))
    snapshot.setdefault("volatility_regime", snapshot.get("volatility_regime", "NORMAL"))
    snapshot.setdefault("order_flow", snapshot.get("order_flow", "NEUTRAL"))
    snapshot.setdefault("spread", snapshot.get("spread", 0.0))
    snapshot.setdefault("rsi", snapshot.get("rsi", 50.0))
    snapshot.setdefault("change_24h", snapshot.get("change_24h", 0.0))
    snapshot.setdefault("volume_24h", snapshot.get("volume_24h", 0.0))
    snapshot.setdefault("distance_to_support", snapshot.get("distance_to_support", 0.0))
    snapshot.setdefault("distance_to_resistance", snapshot.get("distance_to_resistance", 0.0))
    
    # AI/ML
    snapshot.setdefault("ml_direction", snapshot.get("ml_direction", None))
    snapshot.setdefault("ml_confidence", snapshot.get("ml_confidence", 0.0))
    snapshot.setdefault("ml_predicted_move", snapshot.get("ml_predicted_move", 0.0))
    snapshot.setdefault("tft_direction", snapshot.get("tft_direction", None))
    snapshot.setdefault("tft_confidence", snapshot.get("tft_confidence", 0.0))
    snapshot.setdefault("nbeats_direction", snapshot.get("nbeats_direction", None))
    snapshot.setdefault("nbeats_confidence", snapshot.get("nbeats_confidence", 0.0))
    snapshot.setdefault("market_regime", snapshot.get("market_regime", "UNKNOWN"))
    snapshot.setdefault("market_sentiment", snapshot.get("market_sentiment", "NEUTRAL"))
    snapshot.setdefault("confluence_score", snapshot.get("confluence_score", 0.0))
    snapshot.setdefault("confluence_aligned", snapshot.get("confluence_aligned", None))
    
    # Autonomous brain / system state
    snapshot.setdefault("active_pair", snapshot.get("active_pair", None))
    snapshot.setdefault("pair_opportunity_score", snapshot.get("pair_opportunity_score", 0.0))
    snapshot.setdefault("trading_mode", snapshot.get("trading_mode", None))
    snapshot.setdefault("trading_leverage", snapshot.get("trading_leverage", None))
    snapshot.setdefault("trading_side", snapshot.get("trading_side", None))
    snapshot.setdefault("all_modules_initialized", snapshot.get("all_modules_initialized", None))
    snapshot.setdefault("ws_connected", snapshot.get("ws_connected", None))
    
    # Auto-improver
    snapshot.setdefault("auto_improve_total_trades", snapshot.get("auto_improve_total_trades", 0))
    snapshot.setdefault("auto_improve_win_rate", snapshot.get("auto_improve_win_rate", 0.0))
    snapshot.setdefault("auto_improve_consecutive_losses", snapshot.get("auto_improve_consecutive_losses", 0))
    snapshot.setdefault("auto_improve_optimization_count", snapshot.get("auto_improve_optimization_count", 0))
    snapshot.setdefault("auto_optimized_params", snapshot.get("auto_optimized_params", {}))
    
    # Resilience & emergency
    snapshot.setdefault("resilience_healthy", snapshot.get("resilience_healthy", None))
    snapshot.setdefault("resilience_consecutive_failures", snapshot.get("resilience_consecutive_failures", 0))
    snapshot.setdefault("emergency_system", snapshot.get("emergency_system", None))
    
    # Capital profile
    snapshot.setdefault("capital_profile", snapshot.get("capital_profile", "UNKNOWN"))
    snapshot.setdefault("capital_reserve_ratio", snapshot.get("capital_reserve_ratio", 0.0))
    snapshot.setdefault("min_cash_buffer_usdt", snapshot.get("min_cash_buffer_usdt", 0.0))
    snapshot.setdefault("capital_limit_usdt", snapshot.get("capital_limit_usdt", 0.0))
    
    # Risk metrics
    risk_metrics = snapshot.get("risk_metrics", {}) or {}
    snapshot.setdefault("live_trade_risk_fraction", risk_metrics.get("live_trade_risk_fraction", 0.01))
    snapshot.setdefault("signal_quality_score", snapshot.get("signal_quality_score", 
                        risk_metrics.get("setup_quality_score", 0.0)))
    snapshot.setdefault("raw_signal", snapshot.get("raw_signal", snapshot.get("signal", "HOLD")))
    
    # Exit intelligence
    snapshot.setdefault("exit_intelligence_score", snapshot.get("exit_intelligence_score", 0.0))
    snapshot.setdefault("exit_intelligence_threshold", snapshot.get("exit_intelligence_threshold", 0.0))
    snapshot.setdefault("exit_intelligence_reason", snapshot.get("exit_intelligence_reason", None))
    snapshot.setdefault("decision_reason", snapshot.get("decision_reason", None))
    snapshot.setdefault("cooldown_complete", snapshot.get("cooldown_complete", False))
    
    # Strategy signals
    signals = snapshot.get("signals", {}) or {}
    snapshot.setdefault("signal_trend", signals.get("trend", "-"))
    snapshot.setdefault("signal_momentum", signals.get("momentum", "-"))
    snapshot.setdefault("signal_volume", signals.get("volume", "-"))
    snapshot.setdefault("signal_volatility", signals.get("volatility", "-"))
    snapshot.setdefault("signal_structure", signals.get("structure", "-"))
    
    # Emergency & circuit breaker
    snapshot.setdefault("circuit_breaker_trips", snapshot.get("circuit_breaker_trips", 0))
    snapshot.setdefault("emergency_stop_active", snapshot.get("emergency_stop_active", False))
    snapshot.setdefault("exchange_reconnections", snapshot.get("exchange_reconnections", 0))
    snapshot.setdefault("dynamic_exit_enabled", snapshot.get("dynamic_exit_enabled", True))
    
    # Runtime settings
    runtime_settings = snapshot.get("runtime_settings", {}) or {}
    snapshot.setdefault("investment_mode", runtime_settings.get("investment_mode", "Balanced"))
    snapshot.setdefault("optimized_risk_per_trade_fraction", 
                       runtime_settings.get("optimized_risk_per_trade_fraction", 0.01))
    snapshot.setdefault("optimization_reason", runtime_settings.get("optimization_reason", None))
    
    # Health
    snapshot.setdefault("api_latency_p95_ms", snapshot.get("api_latency_p95_ms", 0.0))
    snapshot.setdefault("api_latency_ms", snapshot.get("api_latency_ms", 0.0))
    snapshot.setdefault("ui_render_ms", snapshot.get("ui_render_ms", 0.0))
    snapshot.setdefault("ui_staleness_ms", snapshot.get("ui_staleness_ms", 0.0))
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
                    time.sleep(1.5)
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
                        yield f"data: {json.dumps({'error': str(e), 'retry': 3000})}\n\n"
                    except Exception:
                        break
                    time.sleep(3)

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
        """Send control command to bot."""
        if _global_bot_instance is None:
            return jsonify({"success": False, "error": "Bot not connected"})
        
        try:
            if action == 'pause':
                if hasattr(_global_bot_instance, 'snapshot'):
                    _global_bot_instance.snapshot["cooldown"] = "USER_PAUSED"
                    _global_bot_instance.snapshot["user_paused"] = True
                logger.info("Bot paused via web dashboard")
                return jsonify({"success": True, "message": "Bot paused"})
            elif action == 'resume':
                if hasattr(_global_bot_instance, 'snapshot'):
                    _global_bot_instance.snapshot["cooldown"] = None
                    _global_bot_instance.snapshot["user_paused"] = False
                    _global_bot_instance.snapshot["emergency_stop_active"] = False
                if hasattr(_global_bot_instance, 'emergency_stop_active'):
                    _global_bot_instance.emergency_stop_active = False
                logger.info("Bot resumed via web dashboard")
                return jsonify({"success": True, "message": "Bot resumed"})
            elif action == 'emergency':
                if hasattr(_global_bot_instance, 'snapshot'):
                    _global_bot_instance.snapshot["cooldown"] = "EMERGENCY_STOP"
                    _global_bot_instance.snapshot["emergency_stop_active"] = True
                if hasattr(_global_bot_instance, 'emergency_stop_active'):
                    _global_bot_instance.emergency_stop_active = True
                logger.warning("Emergency stop activated via web dashboard")
                return jsonify({"success": True, "message": "Emergency stop activated"})
            elif action in {'force_close', 'stop_trade'}:
                state_manager = getattr(_global_bot_instance, "state_manager", None)
                if state_manager and hasattr(state_manager, "request_force_close"):
                    state_manager.request_force_close()
                    logger.warning("Manual stop trade requested via web dashboard")
                    return jsonify({"success": True, "message": "Stop Trade request sent"})
                # Usar cola de comandos thread-safe en lugar de llamada directa al event loop del bot
                if hasattr(_global_bot_instance, "state_manager") and hasattr(_global_bot_instance.state_manager, "request_force_close"):
                    _global_bot_instance.state_manager.request_force_close()
                    logger.warning("force_close command queued via web dashboard")
                    return jsonify({"success": True, "message": "Force close command queued"})
                elif hasattr(_global_bot_instance, "request_force_close"):
                    _global_bot_instance.request_force_close()
                    return jsonify({"success": True, "message": "Force close requested"})
                else:
                    return jsonify({"success": False, "error": "Force close not available in this bot version"})
            else:
                return jsonify({"success": False, "error": "Unknown action"})
        except Exception as e:
            logger.error(f"Error sending control: {e}")
            return jsonify({"success": False, "error": str(e)})
    
    @app.route("/api/trades", methods=["GET"])
    @_require_auth
    def api_trades():
        try:
            trades, source = _fetch_trades(limit=200)
            if source == "database":
                trade_rows = []
                for t in trades:
                    trade_rows.append(
                        {
                            "id": str(t.id),
                            "symbol": t.symbol,
                            "side": t.side,
                            "entry_price": float(t.entry_price or 0.0),
                            "current_price": float(t.exit_price or t.entry_price or 0.0),
                            "quantity": float(t.quantity or 0.0),
                            "pnl": float(t.pnl or 0.0),
                            "status": str(t.status or "").lower(),
                            "created_at": (t.timestamp.isoformat() if t.timestamp else datetime.now().isoformat()),
                        }
                    )
                return jsonify({"trades": trade_rows})
            return jsonify({"trades": []})
        except Exception as e:
            logger.error("Error getting trades: %s", e)
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

    logger.info(f"Starting web dashboard on http://{host}:{actual_port}")

    app.run(host=host, port=actual_port, debug=False, use_reloader=False, threaded=True)


def run_in_thread(host: str = '0.0.0.0', port: int = 9000) -> threading.Thread:
    """Run the Flask server in a background thread."""
    thread = threading.Thread(target=run_server, args=(host, port), daemon=True, name="web-dashboard")
    thread.start()
    return thread


if __name__ == '__main__':
    run_server()
