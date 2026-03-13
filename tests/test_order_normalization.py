from __future__ import annotations

from reco_trading.exchange.order_manager import OrderManager, SymbolRules


def _manager_with_rules() -> OrderManager:
    manager = OrderManager(client=None, symbol="BTCUSDT")  # type: ignore[arg-type]
    manager.rules = SymbolRules(
        min_qty=0.0001,
        step_size=0.0001,
        min_notional=10.0,
        tick_size=0.01,
    )
    return manager


def test_quantity_below_min_qty_is_raised_to_minimum() -> None:
    manager = _manager_with_rules()
    qty = manager.normalize_order_quantity(
        symbol="BTCUSDT",
        price=70000,
        quantity=0.00005,
        equity=1000,
        max_trade_balance_fraction=0.2,
    )
    assert qty is not None
    assert qty >= manager.rules.min_qty


def test_step_size_rounding_applies_floor_normalization() -> None:
    manager = _manager_with_rules()
    qty = manager.normalize_order_quantity(
        symbol="BTCUSDT",
        price=1_000_000,
        quantity=0.000123,
        equity=1000,
        max_trade_balance_fraction=0.2,
    )
    assert qty == 0.0001


def test_min_notional_enforcement_adjusts_quantity() -> None:
    manager = _manager_with_rules()
    qty = manager.normalize_order_quantity(
        symbol="BTCUSDT",
        price=70000,
        quantity=0.0001,
        equity=1000,
        max_trade_balance_fraction=0.2,
    )
    assert qty is not None
    assert (70000 * qty) >= manager.rules.min_notional


def test_risk_limit_protection_rejects_if_adjusted_order_exceeds_cap() -> None:
    manager = _manager_with_rules()
    qty = manager.normalize_order_quantity(
        symbol="BTCUSDT",
        price=70000,
        quantity=0.00005,
        equity=20,
        max_trade_balance_fraction=0.2,
    )
    assert qty is None
