from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PredictionSignal:
    direction: str
    confidence: float
    predicted_move_pct: float
    timeframe: str
    factors: dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MarketCondition:
    regime: str
    volatility: float
    trend_strength: float
    volume_profile: str
    market_sentiment: str
    best_direction: str


class EnhancedMLEngine:
    """
    Enhanced ML Engine for high-accuracy predictions.
    Uses multiple models and ensemble voting.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model_ensemble: list[dict] = []
        self.is_trained = False
        
        self.min_confidence_threshold = 0.70
        self.use_ensemble = True
        
        self._price_history: dict[str, list[float]] = {}
        self._volume_history: dict[str, list[float]] = {}
        self._feature_cache: dict[str, dict] = {}

    def add_model(self, model_type: str, weight: float = 1.0) -> None:
        self.model_ensemble.append({
            "type": model_type,
            "weight": weight,
            "accuracy": 0.5
        })
        self.logger.info(f"Added ML model: {model_type} (weight: {weight})")

    async def predict(
        self,
        symbol: str,
        price_data: list[float],
        volume_data: list[float],
        timeframe: str = "5m"
    ) -> PredictionSignal:
        
        if len(price_data) < 20:
            return PredictionSignal(
                direction="HOLD",
                confidence=0.0,
                predicted_move_pct=0.0,
                timeframe=timeframe,
                factors={"error": "Insufficient data"}
            )
        
        self._price_history[symbol] = price_data[-100:]
        self._volume_history[symbol] = volume_data[-100:]
        
        features = self._extract_features(price_data, volume_data)
        self._feature_cache[symbol] = features
        
        signals = []
        
        signal_momentum = self._analyze_momentum(features)
        signals.append(signal_momentum)
        
        signal_trend = self._analyze_trend(features)
        signals.append(signal_trend)
        
        signal_volume = self._analyze_volume(features)
        signals.append(signal_volume)
        
        signal_pattern = self._analyze_patterns(price_data)
        signals.append(signal_pattern)
        
        signal_sentiment = self._analyze_market_sentiment(features)
        signals.append(signal_sentiment)
        
        final_signal = self._ensemble_vote(signals, features)
        
        self.logger.info(
            f"ML Prediction for {symbol}: {final_signal.direction} "
            f"(confidence: {final_signal.confidence:.2%}, move: {final_signal.predicted_move_pct:+.2f}%)"
        )
        
        return final_signal

    def _extract_features(self, prices: list[float], volumes: list[float]) -> dict[str, float]:
        features = {}
        
        returns = np.diff(np.log(prices))
        
        features["returns_mean"] = np.mean(returns)
        features["returns_std"] = np.std(returns)
        features["returns_skew"] = float(np.mean(returns ** 3)) if len(returns) > 0 else 0
        features["returns_kurt"] = float(np.mean(returns ** 4)) if len(returns) > 0 else 0
        
        features["momentum_5"] = (prices[-1] - prices[-6]) / prices[-6] if len(prices) >= 6 else 0
        features["momentum_10"] = (prices[-1] - prices[-11]) / prices[-11] if len(prices) >= 11 else 0
        features["momentum_20"] = (prices[-1] - prices[-21]) / prices[-21] if len(prices) >= 21 else 0
        
        sma5 = np.mean(prices[-5:])
        sma10 = np.mean(prices[-10:])
        sma20 = np.mean(prices[-20:]) if len(prices) >= 20 else sma10
        
        features["sma_5_10_ratio"] = sma5 / sma10 if sma10 != 0 else 1
        features["sma_10_20_ratio"] = sma10 / sma20 if sma20 != 0 else 1
        features["price_sma20_ratio"] = prices[-1] / sma20 if sma20 != 0 else 1
        
        if len(volumes) >= 20:
            avg_volume = np.mean(volumes[-20:])
            features["volume_ratio"] = volumes[-1] / avg_volume if avg_volume > 0 else 1
            features["volume_trend"] = np.mean(np.diff(volumes[-10:]))
        else:
            features["volume_ratio"] = 1
            features["volume_trend"] = 0
        
        if len(returns) >= 14:
            gains = [max(r, 0) for r in returns[-14:]]
            losses = [abs(min(r, 0)) for r in returns[-14:]]
            avg_gain = np.mean(gains) if gains else 0
            avg_loss = np.mean(losses) if losses else 0
            rs = avg_gain / (avg_loss + 1e-10)
            features["rsi"] = 100 - (100 / (1 + rs))
        else:
            features["rsi"] = 50
        
        ema12 = self._calculate_ema(prices, 12)
        ema26 = self._calculate_ema(prices, 26)
        features["macd"] = ema12 - ema26
        features["macd_signal"] = self._calculate_ema(prices[-26:], 9)
        
        features["bb_position"] = self._bollinger_position(prices)
        
        features["volatility_5m"] = np.std(returns[-5:]) if len(returns) >= 5 else 0
        features["volatility_1h"] = np.std(returns[-60:]) if len(returns) >= 60 else 0
        
        return features

    def _calculate_ema(self, prices: list[float], period: int) -> float:
        if len(prices) < period:
            return prices[-1] if prices else 0
        multipliers = [2 / (period + 1) for _ in range(period)]
        weights = [2 ** i for i in range(period)]
        weights.reverse()
        ema = sum(p * w for p, w in zip(prices[-period:], weights)) / sum(weights)
        return ema

    def _bollinger_position(self, prices: list[float], period: int = 20, std_dev: float = 2.0) -> float:
        if len(prices) < period:
            return 0.5
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        if std == 0:
            return 0.5
        position = (prices[-1] - sma) / (std * std_dev)
        return max(0, min(1, (position + 1) / 2))

    def _analyze_momentum(self, features: dict[str, float]) -> dict[str, Any]:
        rsi = features.get("rsi", 50)
        momentum_5 = features.get("momentum_5", 0)
        
        if rsi < 35 and momentum_5 > 0:
            direction = "BUY"
            confidence = min(0.95, 0.6 + (50 - rsi) / 50 + momentum_5 * 5)
        elif rsi > 65 and momentum_5 < 0:
            direction = "SELL"
            confidence = min(0.95, 0.6 + (rsi - 50) / 50 + abs(momentum_5) * 5)
        elif momentum_5 > 0.01:
            direction = "BUY"
            confidence = min(0.7, 0.5 + momentum_5 * 20)
        elif momentum_5 < -0.01:
            direction = "SELL"
            confidence = min(0.7, 0.5 + abs(momentum_5) * 20)
        else:
            direction = "HOLD"
            confidence = 0.3
        
        predicted_move = momentum_5 * 100
        
        return {
            "direction": direction,
            "confidence": confidence,
            "predicted_move": predicted_move,
            "factors": {"rsi": rsi, "momentum": momentum_5}
        }

    def _analyze_trend(self, features: dict[str, float]) -> dict[str, Any]:
        sma_5_10 = features.get("sma_5_10_ratio", 1)
        sma_10_20 = features.get("sma_10_20_ratio", 1)
        price_sma20 = features.get("price_sma20_ratio", 1)
        
        trend_score = 0
        if sma_5_10 > 1.01:
            trend_score += 0.3
        elif sma_5_10 < 0.99:
            trend_score -= 0.3
        
        if sma_10_20 > 1.02:
            trend_score += 0.4
        elif sma_10_20 < 0.98:
            trend_score -= 0.4
        
        if price_sma20 > 1.03:
            trend_score += 0.3
        elif price_sma20 < 0.97:
            trend_score -= 0.3
        
        if trend_score > 0.5:
            direction = "BUY"
            confidence = min(0.85, 0.5 + trend_score * 0.3)
        elif trend_score < -0.5:
            direction = "SELL"
            confidence = min(0.85, 0.5 + abs(trend_score) * 0.3)
        else:
            direction = "HOLD"
            confidence = 0.35
        
        predicted_move = trend_score * 2
        
        return {
            "direction": direction,
            "confidence": confidence,
            "predicted_move": predicted_move,
            "factors": {"sma_ratio": sma_5_10, "trend_score": trend_score}
        }

    def _analyze_volume(self, features: dict[str, float]) -> dict[str, Any]:
        volume_ratio = features.get("volume_ratio", 1)
        volume_trend = features.get("volume_trend", 0)
        
        if volume_ratio > 1.5 and volume_trend > 0:
            direction = "BUY" if features.get("momentum_5", 0) > 0 else "SELL"
            confidence = min(0.75, 0.4 + volume_ratio * 0.2)
        elif volume_ratio < 0.5:
            direction = "HOLD"
            confidence = 0.25
        else:
            direction = "HOLD"
            confidence = 0.3
        
        return {
            "direction": direction,
            "confidence": confidence,
            "predicted_move": 0,
            "factors": {"volume_ratio": volume_ratio}
        }

    def _analyze_patterns(self, prices: list[float]) -> dict[str, Any]:
        if len(prices) < 20:
            return {"direction": "HOLD", "confidence": 0.3, "predicted_move": 0, "factors": {}}
        
        recent = prices[-10:]
        pattern_score = 0
        
        if recent[-1] > recent[-3] > recent[-5] > recent[-7]:
            pattern_score += 0.5
        elif recent[-1] < recent[-3] < recent[-5] < recent[-7]:
            pattern_score -= 0.5
        
        if all(prices[-i] > prices[-i-2] for i in range(1, 4)):
            pattern_score += 0.3
        elif all(prices[-i] < prices[-i-2] for i in range(1, 4)):
            pattern_score -= 0.3
        
        if pattern_score > 0.3:
            direction = "BUY"
            confidence = min(0.7, 0.4 + pattern_score * 0.3)
        elif pattern_score < -0.3:
            direction = "SELL"
            confidence = min(0.7, 0.4 + abs(pattern_score) * 0.3)
        else:
            direction = "HOLD"
            confidence = 0.35
        
        return {
            "direction": direction,
            "confidence": confidence,
            "predicted_move": pattern_score * 1.5,
            "factors": {"pattern_score": pattern_score}
        }

    def _analyze_market_sentiment(self, features: dict[str, float]) -> dict[str, Any]:
        rsi = features.get("rsi", 50)
        volatility = features.get("volatility_5m", 0)
        
        if rsi < 30:
            sentiment = "EXTREME_FEAR"
            direction = "BUY"
            confidence = 0.8
        elif rsi > 70:
            sentiment = "EXTREME_GREED"
            direction = "SELL"
            confidence = 0.8
        elif rsi < 45:
            sentiment = "FEAR"
            direction = "BUY"
            confidence = 0.65
        elif rsi > 55:
            sentiment = "GREED"
            direction = "SELL"
            confidence = 0.65
        else:
            sentiment = "NEUTRAL"
            direction = "HOLD"
            confidence = 0.4
        
        if volatility > 0.03:
            confidence *= 0.8
        
        return {
            "direction": direction,
            "confidence": confidence,
            "predicted_move": (rsi - 50) / 10,
            "factors": {"rsi": rsi, "sentiment": sentiment}
        }

    def _ensemble_vote(self, signals: list[dict], features: dict[str, float]) -> PredictionSignal:
        votes = {"BUY": 0, "SELL": 0, "HOLD": 0}
        total_confidence = {"BUY": 0, "SELL": 0, "HOLD": 0}
        
        for signal in signals:
            direction = signal["direction"]
            confidence = signal["confidence"]
            votes[direction] += 1
            total_confidence[direction] += confidence
        
        final_direction = max(votes, key=votes.get)
        
        if votes[final_direction] >= 3:
            avg_confidence = total_confidence[final_direction] / votes[final_direction]
        else:
            weights = {"BUY": 0.35, "SELL": 0.35, "HOLD": 0.30}
            avg_confidence = sum(
                total_confidence[d] * weights[d]
                for d in total_confidence
            )
        
        predicted_moves = [s["predicted_move"] for s in signals if s["direction"] == final_direction]
        predicted_move = np.mean(predicted_moves) if predicted_moves else 0
        
        all_factors = {}
        for s in signals:
            all_factors.update(s.get("factors", {}))
        
        if avg_confidence < self.min_confidence_threshold:
            final_direction = "HOLD"
            avg_confidence = 0.5
        
        return PredictionSignal(
            direction=final_direction,
            confidence=avg_confidence,
            predicted_move_pct=predicted_move,
            timeframe="5m",
            factors=all_factors
        )

    def get_market_condition(
        self,
        symbol: str,
        prices: list[float],
        volumes: list[float]
    ) -> MarketCondition:
        if len(prices) < 20:
            return MarketCondition(
                regime="UNKNOWN",
                volatility=0,
                trend_strength=0,
                volume_profile="NORMAL",
                market_sentiment="NEUTRAL",
                best_direction="HOLD"
            )
        
        features = self._extract_features(prices, volumes)
        
        volatility = features.get("volatility_5m", 0) * 100
        
        if volatility < 2:
            regime = "LOW_VOLATILITY"
        elif volatility < 5:
            regime = "NORMAL"
        elif volatility < 10:
            regime = "HIGH_VOLATILITY"
        else:
            regime = "EXTREME"
        
        trend_strength = abs(features.get("momentum_5", 0)) * 20
        
        volume_ratio = features.get("volume_ratio", 1)
        if volume_ratio > 1.5:
            volume_profile = "HIGH"
        elif volume_ratio < 0.7:
            volume_profile = "LOW"
        else:
            volume_profile = "NORMAL"
        
        rsi = features.get("rsi", 50)
        if rsi < 30:
            sentiment = "EXTREME_FEAR"
        elif rsi < 45:
            sentiment = "FEAR"
        elif rsi > 70:
            sentiment = "EXTREME_GREED"
        elif rsi > 55:
            sentiment = "GREED"
        else:
            sentiment = "NEUTRAL"
        
        if trend_strength > 0.6 and features.get("momentum_5", 0) > 0:
            best_direction = "BUY"
        elif trend_strength > 0.6 and features.get("momentum_5", 0) < 0:
            best_direction = "SELL"
        else:
            best_direction = "HOLD"
        
        return MarketCondition(
            regime=regime,
            volatility=volatility,
            trend_strength=trend_strength,
            volume_profile=volume_profile,
            market_sentiment=sentiment,
            best_direction=best_direction
        )
