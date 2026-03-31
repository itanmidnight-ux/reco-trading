from __future__ import annotations

import json
import logging
import socket
import threading
import time
from datetime import datetime
from typing import Any, Callable

from flask import Flask, render_template, jsonify, Response


logger = logging.getLogger(__name__)

_global_bot_instance: Any = None
_bot_instance_getter: Callable | None = None
_app: Flask | None = None


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


def get_bot_snapshot() -> dict[str, Any]:
    """Get current snapshot from bot."""
    global _global_bot_instance
    
    if _global_bot_instance is None and _bot_instance_getter is not None:
        try:
            _global_bot_instance = _bot_instance_getter()
        except Exception:
            pass
    
    if _global_bot_instance is None:
        return {
            "status": "WAITING",
            "pair": "BTC/USDT",
            "signal": "HOLD",
            "confidence": 0.0,
            "price": 0.0,
            "daily_pnl": 0.0,
            "equity": 0.0,
            "total_equity": 0.0,
            "win_rate": 0.0,
            "trades_today": 0,
            "has_open_position": False,
            "trend": "NEUTRAL",
            "volatility_regime": "NORMAL",
            "order_flow": "NEUTRAL",
            "adx": 0.0,
            "spread": 0.0,
            "cooldown": "READY",
            "exchange_status": "CONNECTED",
            "database_status": "CONNECTED",
            "trade_history": [],
            "logs": [],
            "timeframe": "5m / 15m",
        }
    
    try:
        snapshot = getattr(_global_bot_instance, 'snapshot', {})
        if callable(snapshot):
            snapshot = snapshot()
        if snapshot is None:
            snapshot = {}
        
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


def create_app() -> Flask:
    """Create and configure Flask app."""
    global _app
    
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.config['JSON_SORT_KEYS'] = False
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/api/snapshot')
    def api_snapshot():
        snapshot = get_bot_snapshot()
        return jsonify(snapshot)
    
    @app.route('/api/status')
    def api_status():
        snapshot = get_bot_snapshot()
        return jsonify({
            "status": snapshot.get("status", "UNKNOWN"),
            "pair": snapshot.get("pair", "N/A"),
            "signal": snapshot.get("signal", "HOLD"),
            "confidence": snapshot.get("confidence", 0),
        })
    
    @app.route('/api/health')
    def api_health():
        return jsonify({
            "status": "running",
            "timestamp": datetime.now().isoformat(),
        })
    
    @app.route('/api/control/<action>', methods=['POST'])
    def api_control(action):
        """Send control command to bot."""
        if _global_bot_instance is None:
            return jsonify({"success": False, "error": "Bot not connected"})
        
        try:
            if action == 'pause':
                _global_bot_instance.snapshot["cooldown"] = "USER_PAUSED"
                _global_bot_instance.snapshot["user_paused"] = True
                logger.info("Bot paused via web dashboard")
                return jsonify({"success": True, "message": "Bot paused"})
            elif action == 'resume':
                _global_bot_instance.snapshot["cooldown"] = None
                _global_bot_instance.snapshot["user_paused"] = False
                _global_bot_instance.snapshot["emergency_stop_active"] = False
                logger.info("Bot resumed via web dashboard")
                return jsonify({"success": True, "message": "Bot resumed"})
            elif action == 'emergency':
                _global_bot_instance.snapshot["cooldown"] = "EMERGENCY_STOP"
                _global_bot_instance.snapshot["emergency_stop_active"] = True
                _global_bot_instance.emergency_stop_active = True
                logger.warning("Emergency stop activated via web dashboard")
                return jsonify({"success": True, "message": "Emergency stop activated"})
            else:
                return jsonify({"success": False, "error": "Unknown action"})
        except Exception as e:
            logger.error(f"Error sending control: {e}")
            return jsonify({"success": False, "error": str(e)})
    
    @app.route('/api/trades')
    def api_trades():
        snapshot = get_bot_snapshot()
        trades = snapshot.get("trade_history", [])
        return jsonify(trades)
    
    @app.route('/api/all_trades')
    def api_all_trades():
        """Get all trades from database."""
        if _global_bot_instance is None:
            return jsonify([])
        
        try:
            repository = getattr(_global_bot_instance, 'repository', None)
            if repository:
                import asyncio
                try:
                    trades = asyncio.get_event_loop().run_until_complete(
                        repository.get_trades(limit=100)
                    )
                    return jsonify([
                        {
                            "trade_id": str(t.trade_id),
                            "pair": t.pair,
                            "side": t.side,
                            "entry": float(t.entry_price) if t.entry_price else 0,
                            "exit": float(t.close_price) if t.close_price else 0,
                            "size": float(t.quantity) if t.quantity else 0,
                            "pnl": float(t.profit_abs) if t.profit_abs else 0,
                            "pnl_percent": float(t.close_rate) * 100 if t.close_rate else 0,
                            "status": t.status,
                            "entry_time": t.entry_timestamp.isoformat() if t.entry_timestamp else None,
                            "exit_time": t.exit_timestamp.isoformat() if t.exit_timestamp else None,
                        }
                        for t in trades
                    ])
                except Exception as e:
                    logger.error(f"Error fetching trades from db: {e}")
            
            snapshot = get_bot_snapshot()
            trades = snapshot.get("trade_history", [])
            return jsonify(trades)
        except Exception as e:
            logger.error(f"Error getting all trades: {e}")
            return jsonify([])
    
    @app.route('/api/analytics')
    def api_analytics():
        snapshot = get_bot_snapshot()
        analytics = snapshot.get("analytics", {})
        return jsonify(analytics)
    
    @app.route('/api/settings')
    def api_settings():
        if _global_bot_instance is None:
            return jsonify({})
        try:
            settings = getattr(_global_bot_instance, 'settings', None)
            if settings:
                return jsonify({
                    "trading_symbol": getattr(settings, 'trading_symbol', 'BTCUSDT'),
                    "timeframe": getattr(settings, 'primary_timeframe', '5m'),
                    "confidence_threshold": getattr(settings, 'confidence_threshold', 0.70),
                })
        except Exception as e:
            logger.error(f"Error getting settings: {e}")
        return jsonify({})
    
    _app = app
    return app


def run_server(host: str = '0.0.0.0', port: int = 9000) -> None:
    """Run the Flask dashboard server."""
    app = create_app()
    
    def find_available_port(start_port: int) -> int:
        """Find an available port starting from start_port."""
        port = start_port
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
