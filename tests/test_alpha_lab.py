import numpy as np
import pandas as pd

from reco_trading.research.alpha_lab import FactorEvaluation, PortfolioBuilder
from reco_trading.research.alpha_lab.factors import LiquidityFactor, MomentumFactor
from reco_trading.research.alpha_lab.walk_forward import AlphaWalkForward


def _sample_frame(rows: int = 400) -> pd.DataFrame:
    idx = pd.date_range('2024-01-01', periods=rows, freq='h')
    base = np.linspace(100, 120, rows)
    return pd.DataFrame(
        {
            'close': base + np.sin(np.linspace(0, 20, rows)),
            'high': base + 1.0,
            'low': base - 1.0,
            'volume': np.linspace(1000, 2000, rows),
            'buy_volume': np.linspace(600, 1200, rows),
            'sell_volume': np.linspace(400, 800, rows),
            'benchmark_close': np.linspace(200, 220, rows) + np.cos(np.linspace(0, 10, rows)),
        },
        index=idx,
    )


def test_alpha_walk_forward_selects_best_factor_and_deploys_signal():
    frame = _sample_frame()
    future_returns = frame['close'].pct_change().shift(-1)
    runner = AlphaWalkForward(
        factors={
            'momentum': MomentumFactor('momentum', lookback=12),
            'liquidity': LiquidityFactor('liquidity', lookback=12),
        },
        train_size=220,
        validate_size=80,
        test_size=60,
    )

    result = runner.run(frame, future_returns)

    assert result.best_factor in {'momentum', 'liquidity'}
    assert set(result.factor_scores) == {'momentum', 'liquidity'}
    assert len(result.deployed_signal) == 60


def test_factor_evaluation_and_portfolio_builder_primitives():
    s1 = pd.Series([0.1, 0.2, -0.3, 0.4, -0.1])
    s2 = pd.Series([0.01, 0.03, -0.02, 0.05, -0.01])
    exposures = pd.DataFrame({'beta': [1.0, 1.1, 0.9, 1.2, 0.95]})

    ic_value = FactorEvaluation.ic(s1, s2)
    turnover = FactorEvaluation.turnover(s1)
    neutral = FactorEvaluation.neutralize(s1, exposures)
    weights = PortfolioBuilder.ic_weighted_allocation({'f1': ic_value, 'f2': 0.2})

    assert isinstance(ic_value, float)
    assert turnover >= 0
    assert neutral.notna().sum() > 0
    assert abs(sum(weights.values()) - 1.0) < 1e-9
