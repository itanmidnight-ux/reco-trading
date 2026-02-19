from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from trading_system.app.backtesting_engine.engine import BacktestReport, BacktestingEngine
from trading_system.app.config.settings import Settings
from trading_system.app.models.ensemble.service import EnsembleService
from trading_system.app.services.decision_engine.service import DecisionEngineService
from trading_system.app.services.feature_engineering.pipeline import FeatureEngineeringService
from trading_system.app.services.market_data.history_builder import OhlcvState
from trading_system.app.services.regime_detection.service import RegimeDetectionService
from trading_system.app.services.risk_management.service import RiskManagementService
from trading_system.app.services.sentiment.service import SentimentSnapshot


@dataclass
class BacktestExecutionConfig:
    warmup_bars: int = 60
    hold_bars: int = 1
    fee_bps: float = 6.0
    slippage_bps: float = 3.0
    latency_ms: float = 120.0
    timeframe_seconds: int = 60
    seed: int = 123


@dataclass
class BacktestRunResult:
    report: BacktestReport
    returns: list[float]
    signals: list[dict[str, Any]]
    trades: list[dict[str, Any]]


class HistoricalBacktestRunner:
    """Runner determinista para backtest usando el mismo stack de servicios de producción."""

    def __init__(
        self,
        settings: Settings,
        config: BacktestExecutionConfig | None = None,
        output_dir: str | Path = 'artifacts/backtests',
    ) -> None:
        self.settings = settings
        self.config = config or BacktestExecutionConfig()
        self.output_dir = Path(output_dir)

        self.feature_engineering = FeatureEngineeringService()
        self.regime_detection = RegimeDetectionService()
        self.ensemble = EnsembleService()
        self.decision_engine = DecisionEngineService()
        self.risk_management = RiskManagementService(settings)
        self.metrics_engine = BacktestingEngine()

    def run(self, candles: list[dict[str, float]], run_id: str = 'default') -> BacktestRunResult:
        if len(candles) <= self.config.warmup_bars + self.config.hold_bars:
            raise ValueError('No hay suficientes velas para ejecutar backtest')

        state = OhlcvState()
        rng = random.Random(self.config.seed)
        signals: list[dict[str, Any]] = []
        trades: list[dict[str, Any]] = []
        returns: list[float] = []

        latency_bars = max(0, round(self.config.latency_ms / max(self.config.timeframe_seconds * 1000, 1)))

        for idx, candle in enumerate(candles):
            state.close.append(float(candle['close']))
            state.high.append(float(candle['high']))
            state.low.append(float(candle['low']))
            state.volume.append(float(candle['volume']))
            state.bid_qty = float(candle.get('bid_qty', candle['volume'] * 0.45))
            state.ask_qty = float(candle.get('ask_qty', candle['volume'] * 0.55))

            if idx < self.config.warmup_bars:
                continue

            entry_idx = idx + latency_bars
            exit_idx = entry_idx + self.config.hold_bars
            if exit_idx >= len(candles):
                break

            features = self.feature_engineering.build(state)
            regime = self.regime_detection.detect(features)
            ensemble = self.ensemble.infer(features, regime)
            sentiment = self._synthetic_sentiment(state)
            decision = self.decision_engine.decide(ensemble, sentiment, features)
            plan = self.risk_management.plan(decision, features, float(candle['close']))

            signal_row = {
                'run_id': run_id,
                'index': idx,
                'timestamp': candle.get('timestamp', idx),
                'close': candle['close'],
                'regime': regime.name,
                'signal': decision.signal,
                'decision_score': decision.score,
                'confidence': decision.confidence,
                'expected_value': decision.expected_value,
                'risk_allow': plan.allow,
                'risk_reason': plan.reason,
            }
            signals.append(signal_row)

            if not plan.allow:
                continue

            entry_candle = candles[entry_idx]
            exit_candle = candles[exit_idx]
            side = 1.0 if decision.signal == 'LONG' else -1.0
            slip = self._slippage_multiplier(rng)
            entry_price = float(entry_candle['open']) * (1.0 + side * slip)
            exit_price = float(exit_candle['close']) * (1.0 - side * slip)
            gross_return = side * ((exit_price - entry_price) / max(entry_price, 1e-9))

            total_cost = (self.config.fee_bps * 2) / 10000
            net_return = gross_return - total_cost
            pnl = plan.qty * net_return
            returns.append(net_return)
            self.risk_management.update(net_return)

            trades.append(
                {
                    'run_id': run_id,
                    'signal_index': idx,
                    'entry_index': entry_idx,
                    'exit_index': exit_idx,
                    'entry_time': entry_candle.get('timestamp', entry_idx),
                    'exit_time': exit_candle.get('timestamp', exit_idx),
                    'side': decision.signal,
                    'qty': plan.qty,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'gross_return': gross_return,
                    'net_return': net_return,
                    'pnl': pnl,
                    'latency_bars': latency_bars,
                    'fees_bps_rt': self.config.fee_bps * 2,
                    'slippage_bps': slip * 10000,
                    'reason': decision.reason,
                }
            )

        self._write_audit_csv(run_id, signals, trades)
        report = self.metrics_engine.run(returns, fee_bps=0.0, slippage_bps=0.0, latency_ms=0.0)
        return BacktestRunResult(report=report, returns=returns, signals=signals, trades=trades)

    def _slippage_multiplier(self, rng: random.Random) -> float:
        jitter = rng.uniform(-0.25, 0.25)
        bps = self.config.slippage_bps * (1.0 + jitter)
        return bps / 10000.0

    @staticmethod
    def _synthetic_sentiment(state: OhlcvState) -> SentimentSnapshot:
        closes = list(state.close)
        if len(closes) < 8:
            return SentimentSnapshot(score=0.0, attention_event=False)
        momentum = (closes[-1] - closes[-8]) / max(closes[-8], 1e-9)
        score = max(-0.65, min(0.65, momentum * 40))
        return SentimentSnapshot(score=score, attention_event=False)

    def _write_audit_csv(self, run_id: str, signals: list[dict[str, Any]], trades: list[dict[str, Any]]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        base = self.output_dir / run_id
        self._write_csv(base.with_name(f'{base.name}_signals.csv'), signals)
        self._write_csv(base.with_name(f'{base.name}_trades.csv'), trades)
        self._write_csv(base.with_name(f'{base.name}_config.csv'), [asdict(self.config)])

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
