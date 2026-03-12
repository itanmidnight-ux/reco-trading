from __future__ import annotations

from reco_trading.exchange.order_manager import OrderManager, SymbolRules
from reco_trading.risk.risk_manager import RiskManager
from reco_trading.strategy.confidence_model import ConfidenceModel
from reco_trading.strategy.signal_engine import SignalBundle


def _bundle() -> SignalBundle:
    return SignalBundle(
        trend="BUY",
        momentum="BUY",
        volume="BUY",
        volatility="BUY",
        structure="BUY",
        order_flow="BUY",
        regime="NORMAL_VOLATILITY",
        regime_trade_allowed=True,
        size_multiplier=1.0,
        atr_ratio=0.01,
        reversal_confirmed=True,
        dip_detected=True,
        liquidity_ok=True,
        support_zone=69000.0,
        resistance_zone=71000.0,
    )


def test_adjust_quantity_for_min_notional_uses_margin() -> None:
    om = OrderManager(client=None, symbol="BTC/USDT")  # type: ignore[arg-type]
    om.rules = SymbolRules(min_qty=0.00000001, step_size=0.00000001, min_notional=10.0, tick_size=0.01)

    price = 70_000.0
    quantity = 0.00005

    adjusted_quantity, was_adjusted = om.adjust_quantity_for_min_notional(quantity, price)

    assert was_adjusted is True
    assert adjusted_quantity > quantity
    assert (adjusted_quantity * price) >= om.rules.min_notional


def test_execution_sizing_pipeline_avoids_notional_below_minimum() -> None:
    bundle = _bundle()
    side, confidence, _ = ConfidenceModel().evaluate(bundle, trade_threshold=0.55)
    assert side == "BUY"
    assert confidence >= 0.55

    risk = RiskManager(0.03, 10)
    sizing = risk.position_size_for_risk(
        equity=350.0,
        risk_fraction=0.01,
        price=70_000.0,
        stop_loss_price=69_000.0,
        atr=250.0,
    )

    om = OrderManager(client=None, symbol="BTC/USDT")  # type: ignore[arg-type]
    om.rules = SymbolRules(min_qty=0.00000001, step_size=0.00000001, min_notional=10.0, tick_size=0.01)

    original_quantity = om.normalize_quantity(sizing.quantity)
    assert (original_quantity * 70_000.0) < 10.0

    adjusted_quantity, was_adjusted = om.adjust_quantity_for_min_notional(original_quantity, 70_000.0)

    assert was_adjusted is True
    assert om.validate_notional(adjusted_quantity, 70_000.0) is True


def test_risk_manager_position_size_regression() -> None:
    sizing = RiskManager(0.03, 10).position_size_for_risk(
        equity=1000.0,
        risk_fraction=0.01,
        price=100.0,
        stop_loss_price=97.0,
        atr=2.0,
    )

    assert round(sizing.risk_amount, 6) == 10.0
    assert round(sizing.stop_distance, 6) == 3.0
    assert round(sizing.quantity, 6) == round((10.0 / 3.0) / 100.0, 6)
