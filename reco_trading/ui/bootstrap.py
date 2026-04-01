from __future__ import annotations

from datetime import datetime
from typing import Any

from reco_trading.database.repository import Repository


def _format_trade(trade: object) -> dict:
    timestamp = getattr(trade, "timestamp", None)
    close_ts = getattr(trade, "close_timestamp", None)
    return {
        "trade_id": getattr(trade, "id", "-"),
        "time": timestamp.isoformat(timespec="seconds") if isinstance(timestamp, datetime) else "-",
        "pair": getattr(trade, "symbol", "-"),
        "side": getattr(trade, "side", "-"),
        "entry": getattr(trade, "entry_price", None),
        "exit": getattr(trade, "exit_price", "-"),
        "size": getattr(trade, "quantity", None),
        "pnl": getattr(trade, "pnl", None),
        "status": getattr(trade, "status", "-"),
        "entry_time": timestamp.isoformat(timespec="seconds") if isinstance(timestamp, datetime) else "-",
        "exit_time": close_ts.isoformat(timespec="seconds") if isinstance(close_ts, datetime) else "-",
        "fees": 0,
        "confidence": None,
        "signal_details": "DB_RESTORED",
    }


def _format_log(log: object) -> dict:
    ts = getattr(log, "timestamp", None)
    return {
        "time": ts.strftime("%H:%M:%S") if isinstance(ts, datetime) else "--:--:--",
        "level": str(getattr(log, "level", "INFO")).upper(),
        "message": str(getattr(log, "message", "")),
    }


async def hydrate_state_from_database(settings: Any, state_manager: object) -> None:
    # Use database fallback: PostgreSQL -> MySQL -> SQLite
    dsn = None
    if settings.postgres_dsn:
        dsn = settings.postgres_dsn
    elif settings.mysql_dsn:
        dsn = settings.mysql_dsn
    elif settings.database_url:
        dsn = settings.database_url
    else:
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        os.makedirs(os.path.join(project_root, "data"), exist_ok=True)
        db_path = os.path.join(project_root, "data", "reco_trading.db")
        dsn = f"sqlite:///{db_path}"
    
    repository = Repository(dsn)
    try:
        await repository.setup()
        trades = await repository.get_recent_trades(limit=200)
        logs = await repository.get_recent_logs(limit=400)
        runtime_bundle = await repository.get_runtime_settings()
        runtime_settings = runtime_bundle.get("ui_runtime_settings", {}) if isinstance(runtime_bundle, dict) else {}

        trade_history = [_format_trade(t) for t in trades]
        log_history = [_format_log(log_item) for log_item in reversed(logs)]
        if hasattr(state_manager, "update"):
            state_manager.update(trade_history=trade_history, logs=log_history, runtime_settings=runtime_settings)
    finally:
        await repository.close()
