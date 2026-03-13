from __future__ import annotations

import pytest

from reco_trading.exchange.order_manager import OrderManager, SymbolRules


@pytest.fixture
def manager() -> OrderManager:
    om = OrderManager(client=None, symbol="BTC/USDT")  # type: ignore[arg-type]
    om.rules = SymbolRules(min_qty=0.0001, step_size=0.0001, min_notional=10.0, tick_size=0.01)
    return om


def test_quantity_below_min_qty_is_raised_to_minimum(manager: OrderManager) -> None:
    normalized = _run_normalize(manager, price=70_000.0, quantity=0.00005, equity=10_000.0, max_fraction=1.0)

    assert normalized is not None
    assert normalized >= 0.0001


def test_step_size_rounding_uses_floor(manager: OrderManager) -> None:
    normalized = _run_normalize(manager, price=200_000.0, quantity=0.000123, equity=10_000.0, max_fraction=1.0)

    assert normalized == 0.0001


def test_min_notional_is_enforced(manager: OrderManager) -> None:
    normalized = _run_normalize(manager, price=70_000.0, quantity=0.0001, equity=10_000.0, max_fraction=1.0)

    assert normalized is not None
    assert 70_000.0 * normalized >= 10.0


def test_returns_none_when_normalized_notional_exceeds_risk_limit(manager: OrderManager) -> None:
    normalized = _run_normalize(manager, price=70_000.0, quantity=0.00005, equity=100.0, max_fraction=0.05)

    assert normalized is None


def _run_normalize(manager: OrderManager, *, price: float, quantity: float, equity: float, max_fraction: float) -> float | None:
    import asyncio

    return asyncio.run(manager.normalize_order_quantity("BTC/USDT", price, quantity, equity, max_fraction))
