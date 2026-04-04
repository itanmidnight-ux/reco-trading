from reco_trading.ui.dashboard import TerminalDashboard


def test_terminal_dashboard_render_never_raises_for_layout_width() -> None:
    dash = TerminalDashboard()
    snapshot = {
        "status": "RUNNING",
        "pair": "BTCUSDT",
        "timeframe": "5m/15m",
        "price": 50000.0,
        "confidence": 0.72,
        "logs": [{"time": "12:00:00", "level": "INFO", "message": "ok"}],
    }
    rendered = dash.render(snapshot)
    # Should return a rich renderable, not the fallback error panel.
    assert "Dashboard render error" not in str(rendered)
