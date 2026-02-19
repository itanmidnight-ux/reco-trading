from trading_system.app.services.monitoring.service import MonitoringService


def test_live_mode_emits_critical_alert() -> None:
    monitoring = MonitoringService()
    monitoring.set_live_mode(True)

    alerts = monitoring.alerts(weight_usage=10, weight_limit=1000, drawdown=0.01, ws_stale_seconds=1)

    assert 'CRITICAL: Sistema operando en LIVE' in alerts
