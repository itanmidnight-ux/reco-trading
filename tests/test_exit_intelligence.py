from __future__ import annotations

import pandas as pd

from reco_trading.core.bot_engine import BotEngine
from reco_trading.risk.position_manager import Position
from reco_trading.strategy.exit_intelligence import ExitIntelligence


def _market_frame_for_buy_fade() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "close": [100.0, 101.2, 102.1, 101.5, 100.9],
            "ema20": [99.9, 100.4, 100.9, 101.0, 100.95],
            "rsi": [62.0, 64.0, 59.0, 54.0, 50.0],
            "macd_diff": [0.20, 0.24, 0.18, 0.08, 0.01],
        }
    )


def test_exit_intelligence_emits_exit_signal_on_giveback_and_momentum_fade() -> None:
    engine = ExitIntelligence()
    position = Position(
        trade_id=1,
        side="BUY",
        quantity=0.10,
        entry_price=100.0,
        stop_loss=98.5,
        take_profit=105.0,
        atr=0.7,
        initial_risk_distance=1.5,
        peak_price=103.0,
        bars_held=24,
    )
    decision = engine.evaluate(
        position=position,
        market_data={"frame5": _market_frame_for_buy_fade(), "spread": 0.12},
        current_price=100.9,
        atr=0.8,
    )
    assert decision.exit_now is True
    assert decision.score >= decision.threshold
    assert decision.reason.startswith("EXIT_INTELLIGENCE_")
    assert decision.reason_codes


def test_exit_intelligence_threshold_changes_with_volatility() -> None:
    engine = ExitIntelligence()
    low_vol = engine._dynamic_threshold(0.003)
    normal_vol = engine._dynamic_threshold(0.006)
    high_vol = engine._dynamic_threshold(0.02)
    assert low_vol < normal_vol < high_vol


def test_position_bars_held_advances_only_on_new_candle_timestamp() -> None:
    bot = BotEngine.__new__(BotEngine)
    position = Position(
        trade_id=2,
        side="BUY",
        quantity=0.1,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=103.0,
        atr=1.0,
    )

    BotEngine._advance_position_bar_count(bot, position, {"candle": {"timestamp": 1000}})
    assert position.bars_held == 1

    BotEngine._advance_position_bar_count(bot, position, {"candle": {"timestamp": 1000}})
    assert position.bars_held == 1

    BotEngine._advance_position_bar_count(bot, position, {"candle": {"timestamp": 2000}})
    assert position.bars_held == 2
