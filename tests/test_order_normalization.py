from __future__ import annotations

from reco_trading.exchange.order_manager import OrderManager, SymbolRules


def _manager(step_size: float = 0.0001) -> OrderManager:
    manager = OrderManager(client=None, symbol="BTC/USDT")  # type: ignore[arg-type]
    manager.rules = SymbolRules(
        min_qty=0.0001,
        step_size=step_size,
        min_notional=10.0,
        tick_size=0.01,
    )
    return manager


def test_quantity_below_min_qty_is_raised_to_minimum() -> None:
    manager = _manager()
    normalized = manager.normalize_order_quantity(
        "BTC/USDT",
        70_000.0,
        0.00005,
        equity=10_000.0,
        max_trade_balance_fraction=1.0,
    )

    assert normalized is not None
    assert normalized >= 0.0001


def test_step_size_rounding_uses_floor() -> None:
    manager = _manager(step_size=0.0001)
    normalized = manager.normalize_order_quantity(
        "BTC/USDT",
        200_000.0,
        0.000123,
        equity=10_000.0,
        max_trade_balance_fraction=1.0,
    )

    assert normalized == 0.0001


def test_min_notional_is_enforced() -> None:
    manager = _manager()
    normalized = manager.normalize_order_quantity(
        "BTC/USDT",
        70_000.0,
        0.0001,
        equity=10_000.0,
        max_trade_balance_fraction=1.0,
    )

    assert normalized is not None
    assert 70_000.0 * normalized >= 10.0


def test_returns_none_when_normalized_notional_exceeds_risk_limit() -> None:
    manager = _manager()
    normalized = manager.normalize_order_quantity(
        "BTC/USDT",
        70_000.0,
        0.00005,
        equity=100.0,
        max_trade_balance_fraction=0.05,
    )

    assert normalized is None
