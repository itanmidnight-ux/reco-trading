from __future__ import annotations

import json
import logging
import socket
import threading
from datetime import datetime
from typing import Any

from flask import Flask, render_template, jsonify, Response


logger = logging.getLogger(__name__)

_global_bot_instance: Any = None
_app: Flask | None = None


def set_bot_instance(bot) -> None:
    """Set the global bot instance for the dashboard to access."""
    global _global_bot_instance
    _global_bot_instance = bot
    logger.info("Bot instance set for web dashboard")


def get_bot_snapshot() -> dict[str, Any]:
    """Get current snapshot from bot."""
    if _global_bot_instance is None:
        return {}
    
    try:
        snapshot = getattr(_global_bot_instance, 'snapshot', {})
        if callable(snapshot):
            snapshot = snapshot()
        return snapshot if snapshot else {}
    except Exception as e:
        logger.error(f"Error getting bot snapshot: {e}")
        return {}


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
    
    @app.route('/api/trades')
    def api_trades():
        snapshot = get_bot_snapshot()
        trades = snapshot.get("trade_history", [])
        return jsonify(trades)
    
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
