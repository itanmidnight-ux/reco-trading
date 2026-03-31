from __future__ import annotations

import logging
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class DriftConfig:
    window_size: int = 100
    alert_threshold: float = 0.05
    severe_threshold: float = 0.15
    warmup_period: int = 50
    check_interval: int = 10


@dataclass
class DriftStatus:
    detected: bool
    severity: str
    drift_magnitude: float
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    details: dict = field(default_factory=dict)


class PerformanceDriftDetector:
    def __init__(self, config: DriftConfig | None = None):
        self.config = config or DriftConfig()
        self.logger = logging.getLogger(__name__)
        
        self._performance_window: deque[float] = deque(maxlen=self.config.window_size)
        self._baseline_performance: float | None = None
        self._drift_history: list[DriftStatus] = []
        
        self._check_counter = 0
        self._is_warmed_up = False
        
        self.logger.info("PerformanceDriftDetector initialized")

    def add_sample(self, performance: float) -> None:
        self._performance_window.append(performance)
        
        if len(self._performance_window) >= self.config.warmup_period and not self._is_warmed_up:
            self._is_warmed_up = True
            self._baseline_performance = np.mean(self._performance_window)
            self.logger.info(f"Baseline established: {self._baseline_performance:.4f}")

    def check_drift(self) -> DriftStatus:
        self._check_counter += 1
        
        if self._check_counter % self.config.check_interval != 0:
            return DriftStatus(
                detected=False,
                severity="none",
                drift_magnitude=0.0,
                confidence=0.0,
                details={"reason": "skip_check"}
            )
        
        if not self._is_warmed_up or self._baseline_performance is None:
            return DriftStatus(
                detected=False,
                severity="warming",
                drift_magnitude=0.0,
                confidence=0.0,
                details={"reason": "not_warmed_up"}
            )
        
        current_perf = np.mean(self._performance_window)
        
        drift_magnitude = abs(current_perf - self._baseline_performance) / abs(self._baseline_performance + 1e-8)
        
        severity = "none"
        if drift_magnitude > self.config.severe_threshold:
            severity = "severe"
        elif drift_magnitude > self.config.alert_threshold:
            severity = "moderate"
        
        detected = drift_magnitude > self.config.alert_threshold
        
        recent_std = np.std(list(self._performance_window)[-20:]) if len(self._performance_window) >= 20 else 0
        confidence = min(1.0, drift_magnitude * 10 / (recent_std + 1))
        
        status = DriftStatus(
            detected=detected,
            severity=severity,
            drift_magnitude=drift_magnitude,
            confidence=confidence,
            details={
                "baseline": self._baseline_performance,
                "current": current_perf,
                "window_size": len(self._performance_window)
            }
        )
        
        if detected:
            self._drift_history.append(status)
            self.logger.warning(f"Drift detected: {severity}, magnitude: {drift_magnitude:.4f}")
        
        return status

    def recalibrate_baseline(self) -> None:
        if len(self._performance_window) >= self.config.warmup_period:
            self._baseline_performance = np.mean(self._performance_window)
            self.logger.info(f"Baseline recalibrated: {self._baseline_performance:.4f}")

    def get_drift_stats(self) -> dict:
        if not self._drift_history:
            return {"total_drifts": 0}
        
        severities = [d.severity for d in self._drift_history]
        
        return {
            "total_drifts": len(self._drift_history),
            "moderate_count": severities.count("moderate"),
            "severe_count": severities.count("severe"),
            "avg_magnitude": np.mean([d.drift_magnitude for d in self._drift_history]),
            "current_baseline": self._baseline_performance,
            "is_warmed_up": self._is_warmed_up
        }


class DataDistributionDetector:
    def __init__(self, window_size: int = 200):
        self.window_size = window_size
        self.logger = logging.getLogger(__name__)
        
        self._reference_stats: dict[str, float] = {}
        self._current_window: deque = deque(maxlen=window_size)

    def set_reference(self, data: np.ndarray) -> None:
        self._reference_stats = {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "median": float(np.median(data)),
            "q25": float(np.percentile(data, 25)),
            "q75": float(np.percentile(data, 75))
        }
        self.logger.info(f"Reference distribution set: mean={self._reference_stats['mean']:.4f}")

    def add_sample(self, value: float) -> None:
        self._current_window.append(value)

    def detect_shift(self) -> dict:
        if not self._reference_stats or len(self._current_window) < 50:
            return {"detected": False, "reason": "insufficient_data"}
        
        current_data = np.array(list(self._current_window))
        
        current_mean = np.mean(current_data)
        current_std = np.std(current_data)
        
        mean_shift = abs(current_mean - self._reference_stats["mean"]) / (self._reference_stats["std"] + 1e-8)
        std_change = abs(current_std - self._reference_stats["std"]) / (self._reference_stats["std"] + 1e-8)
        
        detected = mean_shift > 2.0 or std_change > 2.0
        
        return {
            "detected": detected,
            "mean_shift": mean_shift,
            "std_change": std_change,
            "severity": "severe" if mean_shift > 3 or std_change > 3 else "moderate" if detected else "none"
        }


class ConceptDriftHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self._performance_drift = PerformanceDriftDetector()
        self._data_drift = DataDistributionDetector()
        
        self._adaptation_strategy = "gradual"
        self._consecutive_drifts = 0

    def update(self, performance: float, data: np.ndarray | None = None) -> dict:
        self._performance_drift.add_sample(performance)
        perf_status = self._performance_drift.check_drift()
        
        drift_detected = perf_status.detected
        
        if data is not None:
            self._data_drift.add_sample(float(np.mean(data)))
            data_status = self._data_drift.detect_shift()
            if data_status.get("detected"):
                drift_detected = True
        
        if drift_detected:
            self._consecutive_drifts += 1
            adaptation = self._trigger_adaptation(perf_status)
            return {
                "drift_detected": True,
                "performance_status": {
                    "detected": perf_status.detected,
                    "severity": perf_status.severity,
                    "magnitude": perf_status.drift_magnitude
                },
                "adaptation": adaptation,
                "consecutive_drifts": self._consecutive_drifts
            }
        else:
            if self._consecutive_drifts > 0:
                self._consecutive_drifts = max(0, self._consecutive_drifts - 1)
            
            return {
                "drift_detected": False,
                "consecutive_drifts": self._consecutive_drifts
            }

    def _trigger_adaptation(self, drift_status: DriftStatus) -> dict:
        if drift_status.severity == "severe":
            self.logger.warning("SEVERE drift detected - triggering aggressive adaptation")
            return {
                "strategy": "aggressive",
                "actions": [
                    "recalibrate_baseline",
                    "increase_exploration",
                    "reduce_position_size",
                    "reset_learning_rate"
                ]
            }
        elif drift_status.severity == "moderate":
            self.logger.info("Moderate drift detected - applying gradual adaptation")
            return {
                "strategy": "gradual",
                "actions": [
                    "recalibrate_baseline",
                    "slightly_increase_learning"
                ]
            }
        
        return {"strategy": "none", "actions": []}

    def get_handler_stats(self) -> dict:
        return {
            "performance_drift": self._performance_drift.get_drift_stats(),
            "consecutive_drifts": self._consecutive_drifts,
            "adaptation_strategy": self._adaptation_strategy
        }