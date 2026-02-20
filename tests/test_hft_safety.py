from __future__ import annotations

from typing import Any

import asyncio

from reco_trading.hft.multi_exchange_arbitrage import ExchangeAdapter, MultiExchangeArbitrageEngine
from reco_trading.hft.safety import HFTSafetyMonitor
from reco_trading.monitoring.alert_manager import AlertManager
from reco_trading.monitoring.health_check import HealthCheck


class StubAdapter(ExchangeAdapter):
    def __init__(self, name: str, order_book: dict[str, Any], ticker: dict[str, Any]) -> None:
        self.name = name
        self.order_book = order_book
        self.ticker = ticker

    async def get_order_book(self, symbol: str, limit: int = 20) -> dict[str, Any]:
        return self.order_book

    async def get_ticker(self, symbol: str) -> dict[str, Any]:
        return self.ticker

    async def create_order(self, symbol: str, side: str, amount: float, order_type: str = 'market') -> dict[str, Any]:
        return {'id': f'{self.name}-{side}'}

    async def get_balance(self) -> dict[str, Any]:
        return {}

    async def close(self) -> None:
        return None


def test_safety_monitor_blocks_exchange_on_persistent_desync() -> None:
    monitor = HFTSafetyMonitor(
        alert_manager=AlertManager(),
        book_ticker_desync_bps=5.0,
        anomaly_persistence_threshold=2,
        capital_isolation={'binance': 0.30},
    )

    order_book = {'bids': [[101.0, 1.0]], 'asks': [[103.0, 1.0]]}
    ticker = {'last': 90.0}

    monitor.update_heartbeat('binance')
    monitor.detect_book_ticker_desync('binance', order_book, ticker)
    monitor.detect_book_ticker_desync('binance', order_book, ticker)

    assert 'binance' in monitor.state.blocked_exchanges
    assert monitor.allowed_capital_fraction('binance') == 0.0


def test_engine_respects_safety_auto_disable() -> None:
    monitor = HFTSafetyMonitor(alert_manager=AlertManager())
    monitor.state.auto_disable_arbitrage = True

    engine = MultiExchangeArbitrageEngine(
        adapters={
            'a': StubAdapter('a', {'bids': [[101.0, 1.0]], 'asks': [[102.0, 1.0]]}, {'last': 101.5}),
            'b': StubAdapter('b', {'bids': [[99.0, 1.0]], 'asks': [[100.0, 1.0]]}, {'last': 99.5}),
        },
        min_edge_bps=1.0,
        safety_monitor=monitor,
    )

    opportunities = asyncio.run(engine.scan_symbol('BTC/USDT'))

    assert opportunities == []


def test_hft_health_snapshot_includes_realtime_events() -> None:
    monitor = HFTSafetyMonitor(alert_manager=AlertManager(), anomaly_persistence_threshold=3)
    monitor.update_heartbeat('kraken', heartbeat_ts=0.0)
    monitor.evaluate_heartbeats(now=100.0)

    payload = HealthCheck().hft_safety(monitor)

    assert payload['ok'] is True
    assert payload['hft_safety']['operating_mode'] in {'degraded', 'restricted'}
    assert len(payload['hft_safety']['recent_events']) >= 1
