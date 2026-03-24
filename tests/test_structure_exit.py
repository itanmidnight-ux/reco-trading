from __future__ import annotations

import pandas as pd

from reco_trading.core.bot_engine import BotEngine
from reco_trading.risk.position_manager import Position


def _frame_for_buy_reversal() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "close": [100.0, 101.5, 103.0, 104.2, 105.0, 104.0, 103.2, 102.4, 101.9],
            "high": [100.4, 102.0, 103.6, 104.8, 105.7, 104.6, 103.9, 102.8, 102.2],
            "low": [99.6, 100.9, 102.2, 103.6, 104.4, 103.5, 102.9, 101.8, 101.2],
            "ema20": [100.0, 100.5, 101.2, 102.0, 102.8, 103.1, 103.0, 102.8, 102.6],
        }
    )


def _frame_for_sell_reversal() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "close": [105.0, 104.2, 103.3, 102.5, 101.8, 102.2, 102.7, 103.0, 103.1],
            "high": [105.4, 104.8, 103.9, 103.1, 102.2, 102.6, 103.2, 103.4, 103.5],
            "low": [104.6, 103.7, 102.9, 102.1, 101.2, 101.9, 102.3, 102.7, 102.8],
            "ema20": [104.8, 104.5, 104.0, 103.5, 103.0, 102.9, 102.9, 102.9, 102.95],
        }
    )


def test_detect_structure_exit_for_buy_position() -> None:
    engine = BotEngine.__new__(BotEngine)
    position = Position(
        trade_id=1,
        side="BUY",
        quantity=0.1,
        entry_price=100.0,
        stop_loss=98.0,
        take_profit=110.0,
        atr=1.0,
        peak_price=105.4,
    )
    reason = BotEngine._detect_structure_exit(
        engine,
        position,
        {"frame5": _frame_for_buy_reversal()},
        current_price=101.9,
        atr=1.0,
    )
    assert reason == "STRUCTURE_REVERSAL_EXIT"


def test_detect_structure_exit_for_sell_position() -> None:
    engine = BotEngine.__new__(BotEngine)
    position = Position(
        trade_id=2,
        side="SELL",
        quantity=0.1,
        entry_price=105.0,
        stop_loss=107.0,
        take_profit=95.0,
        atr=1.0,
        peak_price=101.7,
    )
    reason = BotEngine._detect_structure_exit(
        engine,
        position,
        {"frame5": _frame_for_sell_reversal()},
        current_price=103.1,
        atr=1.0,
    )
    assert reason == "STRUCTURE_REVERSAL_EXIT"


def test_structure_exit_requires_two_confirmations() -> None:
    engine = BotEngine.__new__(BotEngine)
    engine.snapshot = {}
    position = Position(
        trade_id=3,
        side="BUY",
        quantity=0.1,
        entry_price=100.0,
        stop_loss=98.0,
        take_profit=110.0,
        atr=1.0,
    )
    first = BotEngine._resolve_structure_exit_signal(engine, position, "STRUCTURE_REVERSAL_EXIT")
    second = BotEngine._resolve_structure_exit_signal(engine, position, "STRUCTURE_REVERSAL_EXIT")
    assert first is None
    assert second == "STRUCTURE_REVERSAL_EXIT"


def test_structure_exit_votes_reset_when_signal_disappears() -> None:
    engine = BotEngine.__new__(BotEngine)
    engine.snapshot = {}
    position = Position(
        trade_id=4,
        side="SELL",
        quantity=0.1,
        entry_price=105.0,
        stop_loss=107.0,
        take_profit=95.0,
        atr=1.0,
    )
    _ = BotEngine._resolve_structure_exit_signal(engine, position, "STRUCTURE_REVERSAL_EXIT")
    none_result = BotEngine._resolve_structure_exit_signal(engine, position, None)
    assert none_result is None
    assert position.structure_exit_votes == 0
