from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass(slots=True)
class AnomalyResult:
    method: str
    score: float
    is_anomaly: bool
    metadata: dict[str, float]


class AnomalyDetectionEngine:
    def __init__(
        self,
        *,
        contamination: float = 0.05,
        random_state: int = 42,
        z_threshold: float = 3.0,
        cp_sigma_factor: float = 4.0,
    ) -> None:
        self.model = IsolationForest(contamination=contamination, random_state=random_state)
        self.z_threshold = max(0.1, z_threshold)
        self.cp_sigma_factor = max(0.5, cp_sigma_factor)

    def isolation_forest(self, values: list[float], *, fit_window: int = 256) -> AnomalyResult:
        data = np.asarray(values, dtype=float)
        if data.size < 10:
            return AnomalyResult('isolation_forest', 0.0, False, {'samples': float(data.size)})

        window = data[-fit_window:]
        reshaped = window.reshape(-1, 1)
        self.model.fit(reshaped)
        score = float(self.model.decision_function([[window[-1]]])[0])
        pred = int(self.model.predict([[window[-1]]])[0])
        return AnomalyResult(
            method='isolation_forest',
            score=score,
            is_anomaly=pred == -1,
            metadata={'samples': float(window.size)},
        )

    def rolling_z_score(self, values: list[float], *, rolling_window: int = 50) -> AnomalyResult:
        data = np.asarray(values, dtype=float)
        if data.size < max(rolling_window, 5):
            return AnomalyResult('rolling_z_score', 0.0, False, {'samples': float(data.size)})

        window = data[-rolling_window:]
        current = float(window[-1])
        mean = float(np.mean(window[:-1]))
        std = float(np.std(window[:-1]))
        if std <= 1e-9:
            z_score = 0.0
        else:
            z_score = (current - mean) / std

        return AnomalyResult(
            method='rolling_z_score',
            score=z_score,
            is_anomaly=abs(z_score) >= self.z_threshold,
            metadata={'window': float(rolling_window), 'mean': mean, 'std': std},
        )

    def change_point_detection(self, values: list[float], *, lookback: int = 120) -> AnomalyResult:
        data = np.asarray(values, dtype=float)
        if data.size < max(lookback, 20):
            return AnomalyResult('change_point', 0.0, False, {'samples': float(data.size)})

        window = data[-lookback:]
        split = lookback // 2
        baseline = window[:split]
        recent = window[split:]
        baseline_mean = float(np.mean(baseline))
        recent_mean = float(np.mean(recent))
        baseline_std = float(np.std(baseline))
        drift = recent_mean - baseline_mean

        threshold = self.cp_sigma_factor * (baseline_std + 1e-9)
        is_anomaly = abs(drift) >= threshold
        normalized = 0.0 if threshold == 0 else drift / threshold

        return AnomalyResult(
            method='change_point',
            score=float(normalized),
            is_anomaly=is_anomaly,
            metadata={
                'baseline_mean': baseline_mean,
                'recent_mean': recent_mean,
                'baseline_std': baseline_std,
                'drift': float(drift),
            },
        )

    def detect(self, values: list[float]) -> list[AnomalyResult]:
        return [
            self.isolation_forest(values),
            self.rolling_z_score(values),
            self.change_point_detection(values),
        ]
