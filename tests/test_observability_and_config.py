from __future__ import annotations

from reco_trading.config.settings import Settings
from reco_trading.core.observability import RuntimeObservability


def test_settings_accepts_alias_and_validates_fractions() -> None:
    settings = Settings(
        postgres_dsn="postgresql+asyncpg://u:p@localhost/db",
        TRADING_SYMBOL="eth/usdt",
        TRADING_SYMBOLS="btcusdt,eth/usdt",
        risk_per_trade_fraction=0.02,
    )
    assert settings.trading_symbol == "ETHUSDT"
    assert settings.trading_symbols == ["BTCUSDT", "ETHUSDT"]


def test_runtime_observability_tracks_p95_and_stale_ratio() -> None:
    obs = RuntimeObservability()
    for latency in [10, 20, 30, 40, 50]:
        obs.record_api_latency(latency)
    obs.record_loop(stale_market_data=True)
    obs.record_loop(stale_market_data=False)
    obs.record_error("exchange")
    obs.record_reconnection()
    obs.record_circuit_breaker_trip()
    obs.update_health(db_healthy=True, exchange_healthy=False)

    snapshot = obs.snapshot()
    assert snapshot["api_latency_p95_ms"] >= 40
    assert 0.49 <= snapshot["stale_market_data_ratio"] <= 0.51
    text = obs.to_prometheus_text()
    assert "reco_api_latency_p95_ms" in text
    assert 'reco_component_errors_total{component="exchange"} 1' in text
