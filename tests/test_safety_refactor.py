from __future__ import annotations

from reco_trading.data.candle_builder import ohlcv_to_frame
from reco_trading.risk.position_manager import Position, PositionManager
from reco_trading.risk.risk_manager import RiskManager
from reco_trading.strategy.confidence_model import ConfidenceModel
from reco_trading.strategy.signal_engine import SignalBundle
from reco_trading.exchange.order_manager import OrderManager


def test_hold_signal_when_confidence_below_threshold() -> None:
    bundle = SignalBundle(
        trend="BUY",
        momentum="NEUTRAL",
        volume="NEUTRAL",
        volatility="NEUTRAL",
        structure="SELL",
        order_flow="NEUTRAL",
        regime="NORMAL_VOLATILITY",
        regime_trade_allowed=True,
        size_multiplier=1.0,
        atr_ratio=0.01,
        reversal_confirmed=False,
        dip_detected=False,
        liquidity_ok=True,
        support_zone=100.0,
        resistance_zone=110.0,
    )
    side, confidence, _ = ConfidenceModel().evaluate(bundle, trade_threshold=0.75)
    assert confidence < 0.75
    assert side == "HOLD"


def test_hold_signal_when_directional_votes_are_tied() -> None:
    bundle = SignalBundle(
        trend="BUY",
        momentum="NEUTRAL",
        volume="BUY",
        volatility="NEUTRAL",
        structure="SELL",
        order_flow="SELL",
        regime="NORMAL_VOLATILITY",
        regime_trade_allowed=True,
        size_multiplier=1.0,
        atr_ratio=0.01,
    )
    side, confidence, _ = ConfidenceModel().evaluate(bundle, trade_threshold=0.0)
    assert confidence > 0.0
    assert side == "HOLD"


def test_risk_position_sizing_uses_stop_distance() -> None:
    sizing = RiskManager(0.03, 10).position_size_for_risk(equity=1000, risk_fraction=0.01, price=100, atr=2)
    assert round(sizing.risk_amount, 6) == 10.0
    assert round(sizing.stop_distance, 6) == 3.0
    assert round(sizing.quantity, 6) == round((10.0 / 3.0) / 100.0, 6)


def test_max_concurrent_trades_enforced() -> None:
    pm = PositionManager()
    pm.open(Position(1, "BUY", 0.1, 100, 95, 110, 2))
    assert pm.can_open(1) is False
    assert pm.can_open(2) is True


def test_candle_validation_filters_invalid_rows() -> None:
    frame = ohlcv_to_frame(
        [
            [2_000, 100, 110, 90, 105, 10],
            [1_000, 100, 95, 90, 105, 10],  # invalid high
            [3_000, 100, 110, 111, 105, 10],  # invalid low
        ]
    )
    assert len(frame) == 1
    assert frame.iloc[0]["timestamp"].value > 0


def test_spread_protection_blocks_wide_spread() -> None:
    om = OrderManager(client=None, symbol="BTC/USDT")  # type: ignore[arg-type]
    assert om.validate_spread(99.0, 101.0, 100.0, max_spread_ratio=0.01) is False


def test_equity_formula() -> None:
    usdt_balance = 150.0
    btc_balance = 0.01
    current_price = 50_000.0
    btc_value = btc_balance * current_price
    total_equity = usdt_balance + btc_value
    assert total_equity == 650.0


def test_snapshot_fields_present() -> None:
    snapshot = {
        "position_side": "BUY",
        "entry_price": 100.0,
        "position_size": 0.2,
        "unrealized_pnl": 2.0,
        "bot_mode": "TESTNET",
        "exchange_status": "CONNECTED",
        "database_status": "CONNECTED",
    }
    expected = {
        "position_side",
        "entry_price",
        "position_size",
        "unrealized_pnl",
        "bot_mode",
        "exchange_status",
        "database_status",
    }
    assert expected.issubset(snapshot.keys())
