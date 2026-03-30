from __future__ import annotations

import asyncio

import pandas as pd

from reco_trading.core.bot_engine import BotEngine
from reco_trading.core.state_machine import BotState
from reco_trading.risk.position_manager import PositionManager
from reco_trading.strategy.confidence_model import ConfidenceModel
from reco_trading.strategy.confluence import ConfluenceResult
from reco_trading.strategy.signal_engine import SignalBundle


class _Settings:
    spot_only_mode = True
    min_signal_confidence = 0.55

    @property
    def confidence_threshold(self) -> float:
        return self.min_signal_confidence


class _SignalEngine:
    def generate(self, df5m: pd.DataFrame, df15m: pd.DataFrame) -> SignalBundle:
        return SignalBundle(
            trend="SELL",
            momentum="SELL",
            volume="NEUTRAL",
            volatility="BUY",
            structure="SELL",
            order_flow="SELL",
            regime="NORMAL_VOLATILITY",
            regime_trade_allowed=True,
            size_multiplier=1.0,
            atr_ratio=0.01,
        )


class _Confluence:
    def evaluate(self, df5m: pd.DataFrame, df15m: pd.DataFrame) -> ConfluenceResult:
        return ConfluenceResult(score=0.9, aligned=True, dominant_side="SELL", notes=["trend_aligned"])


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "open": [100.0] * 30,
        "high": [101.0] * 30,
        "low": [99.0] * 30,
        "close": [100.0] * 30,
        "ema20": [99.0] * 30,
        "ema50": [100.0] * 30,
        "rsi": [40.0] * 30,
        "atr": [1.0] * 30,
        "volume": [1000.0] * 30,
        "vol_ma20": [1000.0] * 30,
        "macd_diff": [-0.1] * 30,
        "stoch_k": [40.0] * 30,
        "adx": [25.0] * 30,
    })


def _engine(*, btc_balance: float) -> BotEngine:
    engine = BotEngine.__new__(BotEngine)
    engine.signal_engine = _SignalEngine()
    engine.confidence_model = ConfidenceModel()
    engine.confluence = _Confluence()
    engine.settings = _Settings()
    engine.position_manager = PositionManager()
    engine.snapshot = {"btc_balance": btc_balance}

    async def _set_state(state: BotState, reason: str | None = None) -> None:
        engine.state = state

    async def _persist_signal(bundle: SignalBundle, side: str, confidence: float) -> None:
        engine._persisted = {"side": side, "confidence": confidence}

    engine._set_state = _set_state
    engine._persist_signal = _persist_signal
    return engine


def test_analyze_market_downgrades_spot_sell_signal_to_hold_without_inventory() -> None:
    engine = _engine(btc_balance=0.0)

    result = asyncio.run(BotEngine.analyze_market(engine, {"frame5": _frame(), "frame15": _frame()}))

    assert result["raw_side"] == "SELL"
    assert result["side"] == "HOLD"
    assert result["confidence"] > 0.70
    assert engine._persisted["side"] == "HOLD"
    assert engine.snapshot["raw_signal"] == "SELL"


def test_analyze_market_keeps_sell_signal_when_inventory_is_available() -> None:
    engine = _engine(btc_balance=0.25)

    result = asyncio.run(BotEngine.analyze_market(engine, {"frame5": _frame(), "frame15": _frame()}))

    assert result["raw_side"] == "SELL"
    assert result["side"] == "SELL"
    assert result["confidence"] > 0.70
    assert engine._persisted["side"] == "SELL"


def test_can_execute_spot_sell_depends_on_inventory() -> None:
    empty_engine = _engine(btc_balance=0.0)
    funded_engine = _engine(btc_balance=0.10)

    assert empty_engine._can_execute_spot_sell() is False
    assert funded_engine._can_execute_spot_sell() is True
