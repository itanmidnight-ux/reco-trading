import numpy as np
import pandas as pd

from reco_trading.research.backtest_engine import BacktestEngine
from reco_trading.research.latency_arbitrage_sim import (
    DistributionSpec,
    LatencyArbitrageSimulator,
    NetworkScenario,
    VenueLatencyConfig,
)


def _simulator() -> LatencyArbitrageSimulator:
    return LatencyArbitrageSimulator(
        [
            VenueLatencyConfig(
                venue="A",
                latency=DistributionSpec("normal", {"mean": 12, "std": 2}),
                jitter=DistributionSpec("normal", {"mean": 1, "std": 1}),
            ),
            VenueLatencyConfig(
                venue="B",
                latency=DistributionSpec("normal", {"mean": 40, "std": 6}),
                jitter=DistributionSpec("normal", {"mean": 4, "std": 2}),
            ),
        ]
    )


def test_rolling_cross_correlation_detects_leadership_lag():
    sim = _simulator()
    n = 500
    idx = pd.RangeIndex(n)
    rng = np.random.default_rng(123)
    leader = pd.Series(rng.normal(0, 1, n), index=idx).cumsum()
    follower = leader.shift(2).fillna(leader.iloc[0])

    rolling = sim.rolling_cross_correlation(leader, follower, window=100, max_lag=4)
    lag = sim.estimate_optimal_lag_by_window(rolling).dropna()

    assert int(round(lag.iloc[300])) == 2


def test_latency_sampling_and_effective_profile_bounds():
    sim = _simulator()
    frame = pd.DataFrame(
        {
            "close": np.linspace(100, 110, 200),
            "high": np.linspace(100.3, 110.6, 200),
            "low": np.linspace(99.8, 109.8, 200),
            "return": np.random.default_rng(7).normal(0, 0.001, 200),
            "volatility": np.random.default_rng(8).uniform(0.001, 0.01, 200),
            "spread": np.random.default_rng(9).uniform(0.0005, 0.005, 200),
            "order_flow_imbalance": np.random.default_rng(10).normal(0, 0.2, 200),
        }
    )

    profile = sim.build_execution_profile(
        frame,
        venue="A",
        scenario=NetworkScenario(name="stress", latency_multiplier=1.4, jitter_multiplier=1.2),
        rng=np.random.default_rng(42),
    )

    assert profile["latency_ms"].min() >= 0.0
    assert profile["slippage_inflation"].min() >= 1.0
    assert profile["execution_probability"].between(0.05, 1.0).all()


def test_sensitivity_report_penalizes_high_latency_scenarios():
    sim = _simulator()
    engine = BacktestEngine(fee_rate=0.0006, slippage_bps=5)

    n = 220
    frame = pd.DataFrame(
        {
            "close": np.linspace(100, 105, n),
            "high": np.linspace(100.4, 105.4, n),
            "low": np.linspace(99.7, 104.8, n),
            "return": np.random.default_rng(11).normal(0.0004, 0.002, n),
            "order_flow_imbalance": np.random.default_rng(12).normal(0.0, 0.25, n),
        }
    )
    signals = pd.Series(np.where(np.arange(n) % 3 == 0, "BUY", "HOLD"))

    scenarios = [
        NetworkScenario(name="fast", latency_multiplier=0.8, jitter_multiplier=0.8, execution_decay=0.03),
        NetworkScenario(name="slow", latency_multiplier=2.5, jitter_multiplier=2.0, execution_decay=0.3),
    ]

    report = sim.sensitivity_report(frame, signals, engine, scenarios=scenarios, venues=["A"], seed=5)
    by_name = {r["scenario"]: r for r in report}

    assert by_name["slow"]["avg_latency_ms"] > by_name["fast"]["avg_latency_ms"]
    assert by_name["slow"]["avg_execution_probability"] < by_name["fast"]["avg_execution_probability"]
    assert by_name["slow"]["terminal_equity"] <= by_name["fast"]["terminal_equity"]
