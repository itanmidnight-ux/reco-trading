"""
Overfitting Detector Module.
Detects overfitting in strategy development.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OverfittingMetrics:
    """Metrics for overfitting detection."""
    train_roi: float
    validation_roi: float
    overfitting_ratio: float
    stability_score: float
    is_overfitting: bool
    confidence: float
    recommendations: list[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class OverfittingDetector:
    """Detects overfitting in strategy performance."""

    def __init__(
        self,
        overfitting_threshold: float = 0.3,
        stability_threshold: float = 0.5,
        min_train_validation_gap: float = 0.15,
    ):
        self.overfitting_threshold = overfitting_threshold
        self.stability_threshold = stability_threshold
        self.min_train_validation_gap = min_train_validation_gap
        
        self._history: list[OverfittingMetrics] = []

    def analyze(
        self,
        train_metrics: dict[str, float],
        validation_metrics: dict[str, float],
        cross_validation_scores: Optional[list[float]] = None,
    ) -> OverfittingMetrics:
        """Analyze for overfitting."""
        train_roi = train_metrics.get("roi", 0)
        validation_roi = validation_metrics.get("roi", 0)
        
        overfitting_ratio = self._calculate_overfitting_ratio(train_roi, validation_roi)
        
        stability_score = self._calculate_stability_score(cross_validation_scores)
        
        is_overfitting = self._detect_overfitting(
            overfitting_ratio, stability_score, train_roi, validation_roi
        )
        
        confidence = self._calculate_confidence(
            overfitting_ratio, stability_score, cross_validation_scores
        )
        
        recommendations = self._generate_recommendations(
            is_overfitting, overfitting_ratio, stability_score, train_roi, validation_roi
        )
        
        metrics = OverfittingMetrics(
            train_roi=train_roi,
            validation_roi=validation_roi,
            overfitting_ratio=overfitting_ratio,
            stability_score=stability_score,
            is_overfitting=is_overfitting,
            confidence=confidence,
            recommendations=recommendations,
        )
        
        self._history.append(metrics)
        
        if len(self._history) > 1000:
            self._history.pop(0)
        
        logger.info(f"Overfitting analysis: ratio={overfitting_ratio:.3f}, stability={stability_score:.3f}, overfitting={is_overfitting}")
        
        return metrics

    def _calculate_overfitting_ratio(self, train_roi: float, validation_roi: float) -> float:
        """Calculate overfitting ratio."""
        if train_roi <= 0:
            return 0.0
        
        gap = (train_roi - validation_roi) / abs(train_roi)
        
        return max(0, gap)

    def _calculate_stability_score(
        self,
        cross_validation_scores: Optional[list[float]],
    ) -> float:
        """Calculate stability score from cross-validation."""
        if not cross_validation_scores or len(cross_validation_scores) < 2:
            return 0.5
        
        scores = np.array(cross_validation_scores)
        
        mean = np.mean(scores)
        std = np.std(scores)
        
        if mean == 0:
            return 0.0
        
        cv = abs(std / mean)
        
        stability = 1 - min(cv, 1.0)
        
        return stability

    def _detect_overfitting(
        self,
        overfitting_ratio: float,
        stability_score: float,
        train_roi: float,
        validation_roi: float,
    ) -> bool:
        """Detect if strategy is overfitting."""
        if overfitting_ratio > self.overfitting_threshold:
            return True
        
        if stability_score < self.stability_threshold:
            return True
        
        if train_roi > 100 and validation_roi < train_roi * 0.5:
            return True
        
        if validation_roi < 0 and train_roi > 0:
            return True
        
        return False

    def _calculate_confidence(
        self,
        overfitting_ratio: float,
        stability_score: float,
        cross_validation_scores: Optional[list[float]],
    ) -> float:
        """Calculate confidence in overfitting detection."""
        confidence = 0.5
        
        if overfitting_ratio > 0.3:
            confidence += 0.2
        elif overfitting_ratio > 0.15:
            confidence += 0.1
        
        confidence += stability_score * 0.3
        
        if cross_validation_scores and len(cross_validation_scores) >= 5:
            confidence += 0.1
        
        return min(confidence, 1.0)

    def _generate_recommendations(
        self,
        is_overfitting: bool,
        overfitting_ratio: float,
        stability_score: float,
        train_roi: float,
        validation_roi: float,
    ) -> list[str]:
        """Generate recommendations to fix overfitting."""
        recommendations = []
        
        if is_overfitting:
            recommendations.append("Strategy appears to be overfitting")
            
            if overfitting_ratio > 0.3:
                recommendations.append("Reduce model complexity or parameter count")
                recommendations.append("Increase regularization")
            
            if stability_score < 0.5:
                recommendations.append("Strategy is unstable across validation sets")
                recommendations.append("Use simpler rules or reduce lookback periods")
            
            if train_roi > 100:
                recommendations.append("Train performance too good - likely overfitting")
                recommendations.append("Use out-of-sample data for final validation")
        
        if validation_roi < 0:
            recommendations.append("Validation ROI is negative - strategy may not work in production")
            recommendations.append("Consider different strategy or market conditions")
        
        if not recommendations:
            recommendations.append("Strategy appears well-validated")
            recommendations.append("Monitor performance in production")
        
        return recommendations

    def analyze_rolling_window(
        self,
        train_results: list[dict[str, float]],
        validation_results: list[dict[str, float]],
    ) -> OverfittingMetrics:
        """Analyze overfitting across rolling windows."""
        if not train_results or not validation_results:
            return OverfittingMetrics(
                train_roi=0,
                validation_roi=0,
                overfitting_ratio=0,
                stability_score=0,
                is_overfitting=False,
                confidence=0,
                recommendations=["Insufficient data for analysis"],
            )
        
        train_rois = [r.get("roi", 0) for r in train_results]
        validation_rois = [r.get("roi", 0) for r in validation_results]
        
        avg_train_roi = sum(train_rois) / len(train_rois)
        avg_validation_roi = sum(validation_rois) / len(validation_rois)
        
        cv_scores = validation_rois
        
        return self.analyze(
            train_metrics={"roi": avg_train_roi},
            validation_metrics={"roi": avg_validation_roi},
            cross_validation_scores=cv_scores,
        )

    def get_current_analysis(self) -> Optional[OverfittingMetrics]:
        """Get current overfitting analysis."""
        if not self._history:
            return None
        return self._history[-1]

    def get_history(self, limit: int = 100) -> list[OverfittingMetrics]:
        """Get overfitting analysis history."""
        return self._history[-limit:]

    def is_stable_over_time(self, window: int = 10) -> bool:
        """Check if strategy is stable over time."""
        if len(self._history) < window:
            return True
        
        recent = self._history[-window:]
        
        overfitting_count = sum(1 for m in recent if m.is_overfitting)
        
        return overfitting_count < window // 3

    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary."""
        current = self.get_current_analysis()
        
        if not current:
            return {"status": "no_data"}
        
        return {
            "train_roi": current.train_roi,
            "validation_roi": current.validation_roi,
            "overfitting_ratio": current.overfitting_ratio,
            "stability_score": current.stability_score,
            "is_overfitting": current.is_overfitting,
            "confidence": current.confidence,
            "recommendations": current.recommendations,
            "timestamp": current.timestamp.isoformat(),
            "is_stable": self.is_stable_over_time(),
        }
