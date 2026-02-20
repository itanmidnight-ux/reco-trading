import asyncio

from reco_trading.hft.multi_exchange_arbitrage import ExecutionConfig, LegPlan, MultiExchangeArbitrageExecutor


class FakeExchange:
    def __init__(self, balances, filled_amount=1.0, submit_delay=0.0, fill_delay=0.0):
        self.balances = balances
        self.filled_amount = filled_amount
        self.submit_delay = submit_delay
        self.fill_delay = fill_delay
        self.cancelled = []
        self.hedges = []

    async def fetch_balance(self):
        return self.balances

    async def create_limit_order(self, symbol, side, amount, price):
        await asyncio.sleep(self.submit_delay)
        return {'id': f'{side}-{symbol}', 'status': 'open'}

    async def wait_for_fill(self, symbol, order_id, timeout_seconds):
        await asyncio.sleep(self.fill_delay)
        return {'status': 'filled', 'filled': self.filled_amount, 'average': 100.0}

    async def cancel_order(self, symbol, order_id):
        self.cancelled.append((symbol, order_id))
        return {'status': 'canceled'}

    async def create_market_order(self, symbol, side, amount):
        self.hedges.append((symbol, side, amount))
        return {'symbol': symbol, 'side': side, 'amount': amount}


def _run(coro):
    return asyncio.run(coro)


def test_execute_two_legs_ok_and_telemetry():
    buy_exchange = FakeExchange({'USDT': {'free': 1_000}, 'BTC': {'free': 0}}, filled_amount=1.0)
    sell_exchange = FakeExchange({'USDT': {'free': 0}, 'BTC': {'free': 2}}, filled_amount=1.0)
    executor = MultiExchangeArbitrageExecutor(
        {'A': buy_exchange, 'B': sell_exchange},
        config=ExecutionConfig(sync_window_ms=50, order_timeout_seconds=1.0),
    )

    report = _run(
        executor.execute_two_leg_arbitrage(
            LegPlan('A', 'BTC/USDT', 'BUY', 1.0, 100.0, 100.0, 'BTC', 'USDT'),
            LegPlan('B', 'BTC/USDT', 'SELL', 1.0, 100.0, 100.0, 'BTC', 'USDT'),
        )
    )

    assert report.success is True
    assert report.buy_leg.telemetry.submit_ts is not None
    assert report.buy_leg.telemetry.ack_ts is not None
    assert report.buy_leg.telemetry.fill_ts is not None
    assert len(executor.latency_telemetry) == 2


def test_pretrade_validation_fails_on_balance():
    buy_exchange = FakeExchange({'USDT': {'free': 10}, 'BTC': {'free': 0}}, filled_amount=1.0)
    sell_exchange = FakeExchange({'USDT': {'free': 0}, 'BTC': {'free': 2}}, filled_amount=1.0)
    executor = MultiExchangeArbitrageExecutor({'A': buy_exchange, 'B': sell_exchange})

    report = _run(
        executor.execute_two_leg_arbitrage(
            LegPlan('A', 'BTC/USDT', 'BUY', 1.0, 100.0, 100.0, 'BTC', 'USDT'),
            LegPlan('B', 'BTC/USDT', 'SELL', 1.0, 100.0, 100.0, 'BTC', 'USDT'),
        )
    )

    assert report.success is False
    assert report.reason == 'pretrade_validation_failed'


def test_contingency_reduces_size_and_locks_pair():
    buy_exchange = FakeExchange({'USDT': {'free': 1_000}, 'BTC': {'free': 0}}, filled_amount=1.0)
    sell_exchange = FakeExchange({'USDT': {'free': 0}, 'BTC': {'free': 2}}, filled_amount=0.4)
    executor = MultiExchangeArbitrageExecutor(
        {'A': buy_exchange, 'B': sell_exchange},
        config=ExecutionConfig(lock_seconds=30, reduction_factor=0.5, min_size_multiplier=0.2),
    )

    report = _run(
        executor.execute_two_leg_arbitrage(
            LegPlan('A', 'BTC/USDT', 'BUY', 1.0, 100.0, 100.0, 'BTC', 'USDT'),
            LegPlan('B', 'BTC/USDT', 'SELL', 1.0, 100.0, 100.0, 'BTC', 'USDT'),
        )
    )

    assert report.success is False
    assert report.reason == 'fill_imbalance_contingency'
    assert report.hedge_order is not None
    assert buy_exchange.hedges or sell_exchange.hedges

    pair_key = tuple(sorted(('A', 'B')))
    assert executor._pair_size_multiplier[pair_key] == 0.5
    assert pair_key in executor._pair_lock_until
