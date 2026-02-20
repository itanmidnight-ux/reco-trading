import asyncio

from reco_trading.execution.execution_firewall import ExecutionFirewall


class _Client:
    def __init__(self, usdt=100_000.0, btc=10.0, depth=100.0, bid=100.0, ask=100.2):
        self.usdt = usdt
        self.btc = btc
        self.depth = depth
        self.bid = bid
        self.ask = ask

    async def fetch_balance(self):
        return {'USDT': {'free': self.usdt}, 'BTC': {'free': self.btc}}

    async def fetch_order_book(self, symbol, limit=20):
        bids = [[self.bid, self.depth / 2], [self.bid * 0.999, self.depth / 2]]
        asks = [[self.ask, self.depth / 2], [self.ask * 1.001, self.depth / 2]]
        return {'bids': bids, 'asks': asks}


def test_execution_firewall_allows_valid_trade():
    fw = ExecutionFirewall(max_slippage_bps=200.0, min_liquidity_coverage=1.0)

    async def _run():
        decision = await fw.evaluate(client=_Client(), symbol='BTC/USDT', side='BUY', amount=1.0)
        assert decision.allowed
        assert decision.reason == 'allowed'
        assert decision.risk_snapshot['symbol'] == 'BTC/USDT'

    asyncio.run(_run())


def test_execution_firewall_rejects_insufficient_balance():
    fw = ExecutionFirewall(max_slippage_bps=200.0, min_liquidity_coverage=1.0)

    async def _run():
        decision = await fw.evaluate(client=_Client(usdt=20.0), symbol='BTC/USDT', side='BUY', amount=1.0)
        assert not decision.allowed
        assert decision.reason == 'insufficient_quote_balance'

    asyncio.run(_run())


def test_execution_firewall_rejects_slippage_limit():
    fw = ExecutionFirewall(max_slippage_bps=1.0, min_liquidity_coverage=1.0)

    async def _run():
        decision = await fw.evaluate(client=_Client(depth=1.0), symbol='BTC/USDT', side='BUY', amount=1.0)
        assert not decision.allowed
        assert decision.reason == 'slippage_limit'

    asyncio.run(_run())
