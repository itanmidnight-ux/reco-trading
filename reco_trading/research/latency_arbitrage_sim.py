from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from reco_trading.research.backtest_engine import BacktestEngine


@dataclass
class DistributionSpec:
    kind: str = "normal"
    params: dict[str, float] = field(default_factory=dict)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        if self.kind == "normal":
            mean = float(self.params.get("mean", 0.0))
            std = max(float(self.params.get("std", 1.0)), 0.0)
            return rng.normal(mean, std, size)
        if self.kind == "lognormal":
            mean = float(self.params.get("mean", 0.0))
            sigma = max(float(self.params.get("sigma", 0.25)), 1e-12)
            return rng.lognormal(mean, sigma, size)
        if self.kind == "uniform":
            low = float(self.params.get("low", 0.0))
            high = float(self.params.get("high", 1.0))
            return rng.uniform(low, high, size)
        raise ValueError(f"Unsupported distribution kind: {self.kind}")


@dataclass
class VenueLatencyConfig:
    venue: str
    latency: DistributionSpec
    jitter: DistributionSpec = field(default_factory=DistributionSpec)


@dataclass
class NetworkScenario:
    name: str
    latency_multiplier: float = 1.0
    jitter_multiplier: float = 1.0
    execution_decay: float = 0.05


class LatencyArbitrageSimulator:
    def __init__(
        self,
        venue_configs: list[VenueLatencyConfig],
        drift_weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
        latency_to_drift_scale: float = 0.0001,
    ) -> None:
        self.venue_configs = {cfg.venue: cfg for cfg in venue_configs}
        self.drift_weights = drift_weights
        self.latency_to_drift_scale = latency_to_drift_scale

    def rolling_cross_correlation(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
        window: int = 50,
        max_lag: int = 5,
    ) -> pd.DataFrame:
        aligned = pd.concat([series_a.rename("a"), series_b.rename("b")], axis=1)
        out = pd.DataFrame(index=aligned.index)
        for lag in range(-max_lag, max_lag + 1):
            out[f"lag_{lag}"] = aligned["a"].rolling(window).corr(aligned["b"].shift(-lag))
        return out

    def estimate_optimal_lag_by_window(self, rolling_corr: pd.DataFrame) -> pd.Series:
        abs_corr = rolling_corr.abs()

        def _best_lag(row: pd.Series) -> float:
            clean = row.dropna()
            if clean.empty:
                return np.nan
            label = clean.idxmax()
            return float(str(label).replace("lag_", ""))

        return abs_corr.apply(_best_lag, axis=1)

    def sample_latency_ms(
        self,
        venue: str,
        size: int,
        rng: np.random.Generator,
        scenario: NetworkScenario | None = None,
    ) -> np.ndarray:
        cfg = self.venue_configs[venue]
        scen = scenario or NetworkScenario(name="baseline")
        base = cfg.latency.sample(rng, size) * scen.latency_multiplier
        jitter = cfg.jitter.sample(rng, size) * scen.jitter_multiplier
        return np.clip(base + jitter, a_min=0.0, a_max=None)

    def compute_latency_drift(
        self,
        volatility: pd.Series,
        spread: pd.Series,
        order_flow_imbalance: pd.Series,
        latency_ms: np.ndarray,
    ) -> pd.Series:
        w_vol, w_spread, w_ofi = self.drift_weights
        component = (
            w_vol * volatility.fillna(volatility.median())
            + w_spread * spread.fillna(spread.median())
            + w_ofi * order_flow_imbalance.fillna(0.0).abs()
        )
        return pd.Series(latency_ms, index=volatility.index) * component * self.latency_to_drift_scale

    def build_execution_profile(
        self,
        frame: pd.DataFrame,
        venue: str,
        scenario: NetworkScenario,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        volatility = frame.get("volatility", frame["return"].rolling(20).std().fillna(0.0))
        spread = frame.get("spread", (frame["high"] - frame["low"]) / frame["close"].replace(0, np.nan))
        ofi = frame.get("order_flow_imbalance", pd.Series(np.zeros(len(frame)), index=frame.index))

        latency_ms = self.sample_latency_ms(venue=venue, size=len(frame), rng=rng, scenario=scenario)
        latency_drift = self.compute_latency_drift(volatility, spread, ofi, latency_ms)
        effective_price = frame["close"] + latency_drift

        execution_probability = np.exp(-scenario.execution_decay * latency_ms / 1000.0)
        slippage_multiplier = 1.0 + (latency_ms / 1000.0) * (1.0 + spread.fillna(spread.median()))

        return pd.DataFrame(
            {
                "venue": venue,
                "scenario": scenario.name,
                "latency_ms": latency_ms,
                "latency_drift": latency_drift,
                "effective_price": effective_price,
                "execution_probability": np.clip(execution_probability, 0.05, 1.0),
                "slippage_inflation": slippage_multiplier.clip(lower=1.0),
            },
            index=frame.index,
        )

    def sensitivity_report(
        self,
        frame: pd.DataFrame,
        signals: pd.Series,
        engine: BacktestEngine,
        scenarios: list[NetworkScenario],
        venues: list[str] | None = None,
        seed: int = 42,
    ) -> list[dict]:
        rng = np.random.default_rng(seed)
        active_venues = venues or list(self.venue_configs.keys())
        metrics = []

        for venue in active_venues:
            for scenario in scenarios:
                profile = self.build_execution_profile(frame, venue, scenario, rng)
                stats = engine.run(
                    frame,
                    signals,
                    latency_slippage_inflation=profile["slippage_inflation"],
                    execution_probability=profile["execution_probability"],
                )
                stats.update(
                    {
                        "venue": venue,
                        "scenario": scenario.name,
                        "avg_latency_ms": float(profile["latency_ms"].mean()),
                        "p95_latency_ms": float(profile["latency_ms"].quantile(0.95)),
                        "avg_execution_probability": float(profile["execution_probability"].mean()),
                    }
                )
                metrics.append(stats)
        return metrics
