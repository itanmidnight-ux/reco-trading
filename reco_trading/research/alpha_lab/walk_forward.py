from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from reco_trading.research.alpha_lab.base import AlphaFactor
from reco_trading.research.alpha_lab.evaluation import FactorEvaluation


@dataclass(slots=True)
class WalkForwardResult:
    best_factor: str
    factor_scores: dict[str, float]
    deployed_signal: pd.Series


class AlphaWalkForward:
    def __init__(self, factors: dict[str, AlphaFactor], train_size: int = 252, validate_size: int = 63, test_size: int = 63) -> None:
        self.factors = factors
        self.train_size = train_size
        self.validate_size = validate_size
        self.test_size = test_size

    def run(self, frame: pd.DataFrame, future_returns: pd.Series) -> WalkForwardResult:
        if len(frame) < self.train_size + self.validate_size + self.test_size:
            raise ValueError('insufficient samples for walk-forward run')

        train_end = self.train_size
        validate_end = train_end + self.validate_size
        test_end = validate_end + self.test_size

        validate_frame = frame.iloc[train_end:validate_end]
        validate_returns = future_returns.iloc[train_end:validate_end]
        test_frame = frame.iloc[validate_end:test_end]

        factor_scores: dict[str, float] = {}
        test_signals: dict[str, pd.Series] = {}

        for name, factor in self.factors.items():
            full_signal = factor.compute(frame)
            factor_scores[name] = FactorEvaluation.ic(full_signal.loc[validate_frame.index], validate_returns)
            test_signals[name] = full_signal.loc[test_frame.index]

        best_factor = max(factor_scores, key=factor_scores.get)
        return WalkForwardResult(
            best_factor=best_factor,
            factor_scores=factor_scores,
            deployed_signal=test_signals[best_factor],
        )


class ResearchFeedbackModule:
    def __init__(self, walk_forward: AlphaWalkForward) -> None:
        self.walk_forward = walk_forward

    def process(self, payload: dict) -> dict:
        frame: pd.DataFrame = payload['research_frame']
        future_returns: pd.Series = payload['research_future_returns']
        result = self.walk_forward.run(frame, future_returns)
        return {
            'best_factor': result.best_factor,
            'factor_scores': result.factor_scores,
            'signal_mean': float(result.deployed_signal.mean()),
        }
