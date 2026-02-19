from __future__ import annotations

from dataclasses import dataclass

from trading_system.app.models.ensemble.service import EnsembleOutput
from trading_system.app.services.feature_engineering.pipeline import FeatureVector
from trading_system.app.services.sentiment.service import SentimentSnapshot


@dataclass
class Decision:
    signal: str
    confidence: float
    score: float
    expected_value: float
    reason: str


class DecisionEngineService:
    def decide(
        self,
        ensemble: EnsembleOutput,
        sentiment: SentimentSnapshot,
        features: FeatureVector,
        ru: float = 1.3,
        rd: float = 1.0,
    ) -> Decision:
        sentiment_norm = (sentiment.score + 1) / 2
        statistical_boost = 0.12 * features.stat_confidence
        penalty_noise = 0.08 if features.volatility > 2.0 else 0.0
        score = max(0.0, min(1.0, 0.80 * ensemble.score + 0.12 * sentiment_norm + statistical_boost - penalty_noise))

        expected_value = ensemble.p_up * ru - ensemble.p_down * rd

        # Gating obligatorio antes de operar: liquidez, volatilidad y confirmación estadística.
        low_liquidity = abs(features.orderbook_imbalance) < 0.01 and features.delta_volume <= 0
        extreme_volatility = features.volatility > 2.8
        weak_statistics = features.stat_pvalue > 0.25 and features.stat_confidence < 0.55
        contradictory_signal = (features.breakout_score > 0 and sentiment.score < -0.6) or (
            features.breakout_score == 0 and sentiment.score > 0.8 and features.hh_hl_lh_ll < 0
        )

        if ensemble.mode == 'SAFE_HOLD':
            signal = 'HOLD'
        elif expected_value <= 0 or low_liquidity or extreme_volatility or weak_statistics or contradictory_signal:
            signal = 'HOLD'
        elif score >= 0.75:
            signal = 'LONG'
        elif score <= 0.25:
            signal = 'SHORT'
        else:
            signal = 'HOLD'

        if sentiment.attention_event and signal != 'HOLD':
            signal = 'HOLD'

        confidence = abs(score - 0.5) * 2
        reason = (
            f'p_up={ensemble.p_up:.3f} score={score:.3f} EV={expected_value:.4f} '
            f'stat_conf={features.stat_confidence:.3f} p={features.stat_pvalue:.3f} '
            f'liq={"low" if low_liquidity else "ok"} vol={features.volatility:.3f} '
            f'mode={ensemble.mode} regime={ensemble.regime} versions={ensemble.model_versions}'
        )
        return Decision(signal=signal, confidence=confidence, score=score, expected_value=expected_value, reason=reason)
