from __future__ import annotations

import asyncio

from reco_trading.core.execution_engine import ExecutionEngine
from reco_trading.kernel.quant_kernel import QuantKernel


class _FakeExchange:
    def __init__(self) -> None:
        self.markets = {
            'BTC/USDT': {
                'limits': {
                    'cost': {'min': 10.0},
                    'amount': {'min': 0.0001},
                }
            }
        }

    async def load_markets(self) -> None:
        return None


class _FakeClient:
    def __init__(self) -> None:
        self.exchange = _FakeExchange()

    async def fetch_ticker(self, symbol: str):
        return {'last': 100.0}


class _FakeDB:
    async def record_order(self, order):
        return None

    async def record_fill(self, fill):
        return None

    async def persist_order_execution(self, execution):
        return None


class _SettingsStub:
    minimal_economic_notional = 8.0


def test_execution_engine_reads_exchange_min_notional() -> None:
    engine = ExecutionEngine(_FakeClient(), 'BTC/USDT', _FakeDB(), redis_url='redis://localhost:6399/0')

    async def _run() -> None:
        min_notional = await engine.get_symbol_min_notional(reference_price=100.0)
        assert min_notional == 10.0

    asyncio.run(_run())


def test_quant_kernel_blocks_when_equity_below_min_notional() -> None:
    kernel = QuantKernel.__new__(QuantKernel)
    kernel.s = _SettingsStub()
    kernel.state = type('State', (), {'equity': 5.0, 'binance_min_notional': 0.0})()
    kernel.execution_engine = type('Exec', (), {'get_symbol_min_notional': staticmethod(lambda reference_price: asyncio.sleep(0, result=10.0))})()

    async def _run() -> None:
        target_notional, reason = await QuantKernel._minimum_order_notional(kernel, last_price=100.0)
        assert target_notional == 10.0
        assert reason == 'insufficient_equity_for_min_notional'
        assert kernel.state.binance_min_notional == 10.0

    asyncio.run(_run())
