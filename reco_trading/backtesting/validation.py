from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from reco_trading.backtesting.engine import BacktestEngine


@dataclass(slots=True)
class FoldReport:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    strategy_return: float
    buy_hold_return: float
    ema_crossover_return: float


@dataclass(slots=True)
class WalkForwardReport:
    folds: list[FoldReport]

    @property
    def consolidated(self) -> dict[str, float]:
        if not self.folds:
            return {"strategy_return": 0.0, "buy_hold_return": 0.0, "ema_crossover_return": 0.0}
        return {
            "strategy_return": sum(f.strategy_return for f in self.folds) / len(self.folds),
            "buy_hold_return": sum(f.buy_hold_return for f in self.folds) / len(self.folds),
            "ema_crossover_return": sum(f.ema_crossover_return for f in self.folds) / len(self.folds),
        }


def run_walk_forward_validation(
    frame5m: pd.DataFrame,
    frame15m: pd.DataFrame,
    *,
    train_window: int = 120,
    test_window: int = 80,
    gap_window: int = 10,
    initial_equity: float = 1000.0,
) -> WalkForwardReport:
    df5 = frame5m.reset_index(drop=True)
    df15 = frame15m.reset_index(drop=True)
    min_len = min(len(df5), len(df15))

    folds: list[FoldReport] = []
    fold_id = 1
    cursor = train_window
    while cursor + gap_window + test_window <= min_len:
        test_start_idx = cursor + gap_window
        test_end_idx = test_start_idx + test_window
        test5 = df5.iloc[test_start_idx:test_end_idx]
        test15 = df15.iloc[test_start_idx:test_end_idx]
        engine = BacktestEngine(initial_equity=initial_equity)
        result = engine.run(test5, test15)

        folds.append(
            FoldReport(
                fold_id=fold_id,
                train_start=pd.to_datetime(df5.iloc[0]["timestamp"], unit="ms", utc=True),
                train_end=pd.to_datetime(df5.iloc[cursor - 1]["timestamp"], unit="ms", utc=True),
                test_start=pd.to_datetime(df5.iloc[test_start_idx]["timestamp"], unit="ms", utc=True),
                test_end=pd.to_datetime(df5.iloc[test_end_idx - 1]["timestamp"], unit="ms", utc=True),
                strategy_return=result.metrics.total_return,
                buy_hold_return=_buy_hold_return(test5),
                ema_crossover_return=_ema_crossover_return(test5),
            )
        )
        fold_id += 1
        cursor += test_window

    return WalkForwardReport(folds=folds)


def _buy_hold_return(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    first = float(frame.iloc[0]["close"])
    last = float(frame.iloc[-1]["close"])
    return (last - first) / max(first, 1e-9)


def _ema_crossover_return(frame: pd.DataFrame) -> float:
    if len(frame) < 30:
        return 0.0
    close = frame["close"].astype(float)
    fast = close.ewm(span=9, adjust=False).mean()
    slow = close.ewm(span=21, adjust=False).mean()
    signal = (fast > slow).astype(int)
    returns = close.pct_change().fillna(0.0)
    strategy_returns = returns * signal.shift(1).fillna(0)
    return float(strategy_returns.sum())
