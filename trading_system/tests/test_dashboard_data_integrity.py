from __future__ import annotations

from trading_system.app.dashboard.service import DashboardService


class _DummyRepository:
    def session(self):  # pragma: no cover
        raise RuntimeError('not used in unit test')


def test_get_metrics_normalizes_seconds_timestamps_and_uses_latest_equity_pnl_total() -> None:
    service = DashboardService(_DummyRepository(), lambda: {})

    async def _trades(limit: int = 1000):
        return [
            {'ts': 1_700_000_000, 'symbol': 'BTCUSDT', 'side': 'BUY', 'qty': 0.01, 'price': 50_000, 'status': 'filled', 'pnl': 10.0},
            {'ts': 1_700_000_100_000, 'symbol': 'BTCUSDT', 'side': 'SELL', 'qty': 0.01, 'price': 50_100, 'status': 'filled', 'pnl': -2.0},
        ]

    async def _equity(limit: int = 2000):
        return [
            {'timestamp': 1_700_000_000_000, 'equity': 120.0, 'drawdown': 0.02, 'pnl_total': 7.5},
            {'timestamp': 1_700_000_100_000, 'equity': 125.0, 'drawdown': 0.05, 'pnl_total': 8.5},
        ]

    service.get_recent_trades = _trades  # type: ignore[method-assign]
    service.get_equity_curve = _equity  # type: ignore[method-assign]

    import asyncio

    metrics = asyncio.run(service.get_metrics())
    assert metrics['capital'] == 125.0
    assert metrics['pnl_total'] == 8.0
    assert metrics['account_pnl_total'] == 8.5
    assert metrics['drawdown'] == 2.0


def test_get_dashboard_payload_prioritizes_real_capital_fields() -> None:
    state = {
        'signal': 'LONG',
        'capital_real_usdt': 321.0,
        'account_equity_usdt': 470.0,
        'risk_active': False,
    }
    service = DashboardService(_DummyRepository(), lambda: state)

    async def _metrics():
        return {
            'capital': 999.0,
            'balance_real': 999.0,
            'pnl_total': 1.0,
            'pnl_daily': 0.5,
            'drawdown': 0.01,
            'total_trades': 1,
            'win_rate': 1.0,
            'expectancy': 1.0,
            'sharpe': 1.0,
            'equity_curve': [],
            'losses': 0,
        }

    service.get_metrics = _metrics  # type: ignore[method-assign]

    import asyncio

    payload = asyncio.run(service.get_dashboard_payload())
    assert payload['capital'] == 321.0
    assert payload['balance_real'] == 321.0
    assert payload['capital_real_usdt'] == 321.0
    assert payload['account_equity_usdt'] == 470.0


def test_get_activity_feed_merges_signals_and_executions_sorted() -> None:
    service = DashboardService(_DummyRepository(), lambda: {})

    async def _trades(limit: int = 100):
        return [
            {'ts': 1700000001000, 'symbol': 'BTCUSDT', 'side': 'BUY', 'qty': 0.01, 'price': 50000.0, 'status': 'entry_filled', 'pnl': 0.0},
        ]

    class _Signal:
        ts = 1700000002000
        signal = 'LONG'
        symbol = 'BTCUSDT'
        score = 0.88
        expected_value = 0.0123
        reason = 'test_reason'

    async def _fetch_all(stmt):
        return [_Signal()]

    service.get_recent_trades = _trades  # type: ignore[method-assign]
    service._fetch_all = _fetch_all  # type: ignore[method-assign]

    import asyncio

    feed = asyncio.run(service.get_activity_feed(limit=10))
    assert len(feed) == 2
    assert feed[0]['type'] == 'signal'
    assert feed[1]['type'] == 'execution'
