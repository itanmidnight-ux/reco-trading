from reco_trading.monitoring.alert_manager import AlertManager


def test_alert_manager_evaluates_slo_alerts(monkeypatch):
    manager = AlertManager()
    emitted: list[tuple[str, str, str]] = []

    def _fake_emit(title: str, detail: str, *, severity: str = 'error', exchange=None, payload=None):
        emitted.append((title, severity, detail))

    monkeypatch.setattr(manager, 'emit', _fake_emit)

    manager.evaluate_slo_alerts(
        error_rate=0.01,
        p95_latency_seconds=0.08,
        fill_ratio=0.85,
        drawdown_ratio=0.15,
        capital_protection_active=True,
        exchange='binance',
    )

    names = {name for name, _severity, _detail in emitted}
    assert 'error budget burn' in names
    assert 'latency breach' in names
    assert 'execution anomaly' in names
    assert 'capital protection active' in names
