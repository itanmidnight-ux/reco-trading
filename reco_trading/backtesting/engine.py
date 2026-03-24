from __future__ import annotations

from dataclasses import dataclass
import logging

import pandas as pd

from reco_trading.backtesting.data_loader import BacktestDataLoader
from reco_trading.backtesting.performance_metrics import PerformanceMetrics, compute_metrics
from reco_trading.backtesting.simulator import TradeSimulator, SimulatedTrade
from reco_trading.strategy.confidence_model import ConfidenceModel
from reco_trading.strategy.indicators import apply_indicators
from reco_trading.strategy.signal_engine import SignalEngine


@dataclass(slots=True)
class BacktestResult:
    trades: list[SimulatedTrade]
    metrics: PerformanceMetrics
    equity_curve: list[float]


class BacktestEngine:
    """Historical replay engine that reuses live strategy components safely offline."""

    def __init__(
        self,
        initial_equity: float = 1000.0,
        risk_fraction: float = 0.01,
        maker_fee_rate: float = 0.0002,
        taker_fee_rate: float = 0.0007,
    ) -> None:
        self.initial_equity = initial_equity
        self.risk_fraction = risk_fraction
        self.logger = logging.getLogger(__name__)
        self.loader = BacktestDataLoader()
        self.signal_engine = SignalEngine()
        self.confidence_model = ConfidenceModel()
        self.maker_fee_rate = maker_fee_rate
        self.taker_fee_rate = taker_fee_rate

    def run(self, frame5m: pd.DataFrame, frame15m: pd.DataFrame, confidence_threshold: float = 0.75) -> BacktestResult:
        base5 = self.loader.from_dataframe(frame5m, "5m").frame
        base15 = self.loader.from_dataframe(frame15m, "15m").frame
        df5 = apply_indicators(base5)
        df15 = apply_indicators(base15)

        simulator = TradeSimulator(maker_fee_rate=self.maker_fee_rate, taker_fee_rate=self.taker_fee_rate)
        equity = float(self.initial_equity)
        equity_curve = [equity]

        min_len = min(len(df5), len(df15))
        for idx in range(60, min_len):
            w5 = df5.iloc[: idx + 1]
            w15 = df15.iloc[: idx + 1]
            bundle = self.signal_engine.generate(w5, w15)
            side, confidence, _ = self.confidence_model.evaluate(bundle, confidence_threshold)
            row = w5.iloc[-1]
            price = float(row["close"])
            high = float(row["high"])
            low = float(row["low"])
            volume = float(row.get("volume", 0.0))
            range_ratio = (high - low) / max(price, 1e-9)
            liquidity_ratio = min(max(volume / 1000.0, 0.05), 1.0)
            timestamp = row["timestamp"].to_pydatetime()

            if side == "HOLD":
                if simulator.open_trade:
                    closed = simulator.close_position(
                        price,
                        timestamp,
                        reason="HOLD_EXIT",
                        volatility_ratio=range_ratio,
                        liquidity_ratio=liquidity_ratio,
                    )
                    if closed:
                        equity += closed.pnl
                equity_curve.append(equity)
                continue

            qty = max((equity * self.risk_fraction) / max(price * 0.01, 1e-9), 0.0)
            if simulator.open_trade and simulator.open_trade.side != side:
                closed = simulator.close_position(
                    price,
                    timestamp,
                    reason="SIGNAL_FLIP",
                    volatility_ratio=range_ratio,
                    liquidity_ratio=liquidity_ratio,
                )
                if closed:
                    equity += closed.pnl
            if simulator.open_trade is None:
                simulator.open_position(
                    side=side,
                    quantity=qty,
                    price=price,
                    timestamp=timestamp,
                    volatility_ratio=range_ratio,
                    liquidity_ratio=liquidity_ratio,
                )
            equity_curve.append(equity)

        if simulator.open_trade and len(df5):
            last = df5.iloc[-1]
            last_price = float(last["close"])
            last_range_ratio = (float(last["high"]) - float(last["low"])) / max(last_price, 1e-9)
            closed = simulator.close_position(
                last_price,
                last["timestamp"].to_pydatetime(),
                reason="END_OF_BACKTEST",
                volatility_ratio=last_range_ratio,
                liquidity_ratio=min(max(float(last.get("volume", 0.0)) / 1000.0, 0.05), 1.0),
            )
            if closed:
                equity += closed.pnl
                equity_curve.append(equity)

        metrics = compute_metrics(simulator.trades, self.initial_equity, equity_curve)
        return BacktestResult(trades=simulator.trades, metrics=metrics, equity_curve=equity_curve)
