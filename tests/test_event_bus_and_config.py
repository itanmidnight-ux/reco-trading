from __future__ import annotations

from reco_trading.config.config_loader import ConfigLoader
from reco_trading.core.event_bus import EventBus


def test_event_bus_publish_subscribe() -> None:
    bus = EventBus()
    seen: list[int] = []
    bus.subscribe("metrics", lambda e: seen.append(int(e.payload["v"])))
    bus.publish("metrics", {"v": 7})
    processed = bus.drain()
    assert processed == 1
    assert seen == [7]


def test_config_loader_loads_defaults_and_files() -> None:
    bundle = ConfigLoader(base_dir="config").load()
    assert bundle.trading["symbol"]
    assert bundle.risk["risk_per_trade_fraction"] >= 0
    assert 0 <= bundle.strategy["min_signal_confidence"] <= 1
