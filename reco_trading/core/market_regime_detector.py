from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MarketRegime:
    regime: str
    confidence: float
    volatility: float
    trend: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    features: dict = field(default_factory=dict)


class HiddenMarkovModel:
    """
    Hidden Markov Model for market regime detection.
    Detects: BULL, BEAR, SIDEWAYS, HIGH_VOLATILITY, LOW_VOLATILITY
    """

    def __init__(self, n_states: int = 5):
        self.logger = logging.getLogger(__name__)
        self.n_states = n_states
        
        self._transition_matrix = np.zeros((n_states, n_states))
        self._emission_means = np.zeros(n_states)
        self._emission_stds = np.zeros(n_states)
        self._initial_probs = np.ones(n_states) / n_states
        
        self._current_state = 0
        self._state_history: list[int] = []
        
        self._fitted = False

    def fit(self, returns: list[float], n_iter: int = 50) -> None:
        """Fit HMM to return data using Baum-Welch approximation."""
        
        if len(returns) < 20:
            self.logger.warning("Not enough data for HMM fitting")
            return
        
        returns_array = np.array(returns)
        
        self._emission_means = np.linspace(
            returns_array.min(),
            returns_array.max(),
            self.n_states
        )
        self._emission_stds = np.ones(self.n_states) * returns_array.std()
        
        self._transition_matrix = np.eye(self.n_states) * 0.7
        self._transition_matrix += 0.3 / self.n_states
        
        for iteration in range(n_iter):
            self._em_step(returns_array)
        
        self._fitted = True
        self.logger.info(f"HMM fitted with {self.n_states} states")

    def _em_step(self, returns: np.ndarray) -> None:
        """EM step approximation."""
        
        log_likelihoods = np.zeros((len(returns), self.n_states))
        for state in range(self.n_states):
            log_likelihoods[:, state] = self._gaussian_pdf(
                returns,
                self._emission_means[state],
                self._emission_stds[state]
            )
        
        forward_probs = self._forward_pass(log_likelihoods)
        backward_probs = self._backward_pass(log_likelihoods)
        
        gamma = forward_probs * backward_probs
        gamma = gamma / gamma.sum(axis=1, keepdims=True)
        
        xi = self._compute_xi(forward_probs, backward_probs, log_likelihoods)
        
        for i in range(self.n_states):
            self._emission_means[i] = np.sum(gamma[:, i] * returns) / (np.sum(gamma[:, i]) + 1e-10)
            std = np.sqrt(np.sum(gamma[:, i] * (returns - self._emission_means[i])**2) / (np.sum(gamma[:, i]) + 1e-10))
            self._emission_stds[i] = max(std, 1e-6)
        
        self._transition_matrix = xi.sum(axis=0) / (xi.sum(axis=0).sum(axis=1, keepdims=True) + 1e-10)
        
        self._initial_probs = gamma[0] / gamma[0].sum()

    def _gaussian_pdf(self, x: np.ndarray, mean: float, std: float) -> np.ndarray:
        """Gaussian probability density."""
        return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * math.sqrt(2 * math.pi))

    def _forward_pass(self, log_likelihoods: np.ndarray) -> np.ndarray:
        """Forward pass with numerical stability."""
        n_obs = log_likelihoods.shape[0]
        forward = np.zeros((n_obs, self.n_states))
        
        log_initial = np.log(self._initial_probs + 1e-300)
        log_likelihoods = np.clip(log_likelihoods, -100, 100)
        
        forward[0] = np.exp(log_initial + log_likelihoods[0])
        forward[0] /= forward[0].sum() + 1e-10
        
        for t in range(1, n_obs):
            log_transition = np.log(self._transition_matrix + 1e-300)
            log_sum = np.log(forward[t-1] + 1e-300).reshape(-1, 1) + log_transition
            log_alpha = np.log(np.sum(np.exp(log_sum - log_sum.max()), axis=0, keepdims=True)) + log_sum.max()
            forward[t] = np.exp(log_alpha.flatten() + log_likelihoods[t])
            forward[t] = np.clip(forward[t], 0, 1e10)
            forward[t] /= forward[t].sum() + 1e-10
        
        return forward

    def _backward_pass(self, log_likelihoods: np.ndarray) -> np.ndarray:
        """Backward pass with numerical stability."""
        n_obs = log_likelihoods.shape[0]
        backward = np.zeros((n_obs, self.n_states))
        
        backward[-1] = 1.0
        log_likelihoods = np.clip(log_likelihoods, -100, 100)
        
        for t in range(n_obs - 2, -1, -1):
            log_transition = np.log(self._transition_matrix + 1e-300)
            log_beta_next = np.log(backward[t+1] + 1e-300)
            log_obs = log_likelihoods[t+1]
            log_sum = log_transition + log_obs.reshape(1, -1) + log_beta_next.reshape(1, -1)
            log_beta = np.log(np.sum(np.exp(log_sum - log_sum.max()), axis=1, keepdims=True)) + log_sum.max()
            backward[t] = np.exp(log_beta.flatten())
            backward[t] = np.clip(backward[t], 0, 1e10)
            backward[t] /= backward[t].sum() + 1e-10
        
        return backward

    def _compute_xi(self, forward: np.ndarray, backward: np.ndarray, log_likelihoods: np.ndarray) -> np.ndarray:
        """Compute xi for transition matrix update."""
        n_obs = forward.shape[0]
        xi = np.zeros((n_obs - 1, self.n_states, self.n_states))
        log_likelihoods = np.clip(log_likelihoods, -100, 100)
        
        for t in range(n_obs - 1):
            log_transition = np.log(self._transition_matrix + 1e-300)
            log_obs = log_likelihoods[t+1]
            log_beta = np.log(backward[t+1] + 1e-300)
            
            log_numerator = (np.log(forward[t] + 1e-300).reshape(-1, 1) + 
                           log_transition + 
                           log_obs.reshape(1, -1) + 
                           log_beta.reshape(1, -1))
            
            log_numerator = log_numerator - log_numerator.max()
            numerator = np.exp(log_numerator)
            denominator = numerator.sum() + 1e-10
            xi[t] = numerator / denominator
        
        return xi

    def predict_state(self, returns: list[float]) -> int:
        """Predict most likely state for given returns."""
        
        if not self._fitted or len(returns) < 10:
            return self._estimate_simple_state(returns)
        
        returns_array = np.array(returns)
        
        log_likelihoods = np.zeros((len(returns), self.n_states))
        for state in range(self.n_states):
            log_likelihoods[:, state] = self._gaussian_pdf(
                returns_array,
                self._emission_means[state],
                self._emission_stds[state]
            )
        
        forward = self._forward_pass(log_likelihoods)
        
        state_probs = forward[-1]
        predicted_state = int(np.argmax(state_probs))
        
        self._current_state = predicted_state
        self._state_history.append(predicted_state)
        
        return predicted_state

    def _estimate_simple_state(self, returns: list[float]) -> int:
        """Simple state estimation when not fitted."""
        
        if not returns:
            return 0
        
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        volatility = np.std(returns_array)
        
        if volatility > 0.03:
            state = 4
        elif mean_return > 0.005:
            state = 0
        elif mean_return < -0.005:
            state = 2
        else:
            state = 1
        
        self._current_state = state
        self._state_history.append(state)
        
        return state

    def get_state_name(self, state: int | None = None) -> str:
        """Get regime name for state."""
        
        s = state if state is not None else self._current_state
        
        regime_names = {
            0: "BULL",
            1: "SIDEWAYS",
            2: "BEAR",
            3: "LOW_VOLATILITY",
            4: "HIGH_VOLATILITY",
        }
        
        return regime_names.get(s, "UNKNOWN")


class MarketRegimeDetector:
    """
    Complete market regime detection system.
    Uses HMM + technical indicators for robust regime detection.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self.hmm = HiddenMarkovModel(n_states=5)
        self._regime_history: list[MarketRegime] = []
        self._price_history: list[float] = []
        self._return_history: list[float] = []
        
        self._regime_names = {
            0: "BULL",
            1: "SIDEWAYS", 
            2: "BEAR",
            3: "LOW_VOLATILITY",
            4: "HIGH_VOLATILITY",
        }

    def update(self, prices: list[float], volumes: list[float] | None = None) -> MarketRegime:
        """Update regime detection with new price data."""
        
        self._price_history.extend(prices)
        
        if len(self._price_history) > 500:
            self._price_history = self._price_history[-300:]
        
        if len(self._price_history) < 2:
            return MarketRegime("UNKNOWN", 0.0, 0.0, "NEUTRAL")
        
        returns = []
        for i in range(1, len(self._price_history)):
            if self._price_history[i-1] != 0:
                ret = (self._price_history[i] - self._price_history[i-1]) / self._price_history[i-1]
                returns.append(ret)
        
        self._return_history.extend(returns)
        
        if len(self._return_history) > 500:
            self._return_history = self._return_history[-300:]
        
        return self._detect_regime(returns, volumes)

    def _detect_regime(self, returns: list[float], volumes: list[float] | None) -> MarketRegime:
        """Detect current market regime."""
        
        if len(returns) < 20:
            return MarketRegime("UNKNOWN", 0.0, 0.0, "NEUTRAL")
        
        if not self.hmm._fitted:
            try:
                self.hmm.fit(returns[-60:])
            except Exception as e:
                self.logger.error(f"HMM fit error: {e}")
        
        hmm_state = self.hmm.predict_state(returns[-30:])
        regime_name = self._regime_names.get(hmm_state, "UNKNOWN")
        
        volatility = float(np.std(returns[-30:]))
        
        mean_return = float(np.mean(returns[-30:]))
        
        if mean_return > 0.003:
            trend = "BULLISH"
        elif mean_return < -0.003:
            trend = "BEARISH"
        else:
            trend = "SIDEWAYS"
        
        if volatility > 0.04:
            regime_name = "HIGH_VOLATILITY"
        elif volatility < 0.01:
            regime_name = "LOW_VOLATILITY"
        
        if volumes and len(volumes) >= 20:
            avg_volume = np.mean(volumes[-30:])
            recent_volume = np.mean(volumes[-5:])
            if recent_volume > avg_volume * 1.5:
                regime_name = "HIGH_VOLATILITY"
        
        confidence = min(1.0, len(returns) / 100)
        
        regime = MarketRegime(
            regime=regime_name,
            confidence=confidence,
            volatility=volatility,
            trend=trend,
            features={
                "mean_return": mean_return,
                "hmm_state": hmm_state,
                "price_change_5d": (self._price_history[-1] / self._price_history[-6] - 1) if len(self._price_history) > 5 else 0,
                "price_change_20d": (self._price_history[-1] / self._price_history[-21] - 1) if len(self._price_history) > 20 else 0,
            }
        )
        
        self._regime_history.append(regime)
        
        if len(self._regime_history) > 100:
            self._regime_history = self._regime_history[-50:]
        
        return regime

    def get_current_regime(self) -> MarketRegime:
        """Get current market regime."""
        
        if self._regime_history:
            return self._regime_history[-1]
        
        return MarketRegime("UNKNOWN", 0.0, 0.0, "NEUTRAL")

    def get_regime_statistics(self) -> dict[str, Any]:
        """Get regime statistics."""
        
        if not self._regime_history:
            return {"current": "UNKNOWN", "history": []}
        
        regime_counts = {}
        for regime in self._regime_history:
            regime_counts[regime.regime] = regime_counts.get(regime.regime, 0) + 1
        
        return {
            "current": self._regime_history[-1].regime,
            "current_trend": self._regime_history[-1].trend,
            "current_volatility": self._regime_history[-1].volatility,
            "regime_distribution": regime_counts,
            "history_length": len(self._regime_history),
        }

    def get_recommended_filters(self) -> dict[str, Any]:
        """Get recommended filters based on current regime."""
        
        regime = self.get_current_regime()
        
        filters = {
            "min_signal_confidence": 0.70,
            "stop_loss": 2.0,
            "take_profit": 4.0,
            "position_size": 5.0,
        }
        
        if regime.regime == "HIGH_VOLATILITY":
            filters["min_signal_confidence"] = 0.50
            filters["stop_loss"] = 3.0
            filters["take_profit"] = 6.0
            filters["position_size"] = 3.0
        elif regime.regime == "LOW_VOLATILITY":
            filters["min_signal_confidence"] = 0.30
            filters["stop_loss"] = 1.5
            filters["take_profit"] = 3.0
            filters["position_size"] = 7.0
        elif regime.regime == "BULL":
            filters["min_signal_confidence"] = 0.35
            filters["position_size"] = 6.0
        elif regime.regime == "BEAR":
            filters["min_signal_confidence"] = 0.45
            filters["position_size"] = 3.0
        elif regime.regime == "SIDEWAYS":
            filters["min_signal_confidence"] = 0.25
            filters["take_profit"] = 3.0
            filters["stop_loss"] = 1.5
        
        return filters


class AdaptiveFilter:
    """Base class for adaptive filters."""

    def __init__(self, name: str):
        self.name = name
        self._values: list[float] = []

    def update(self, value: float) -> float:
        raise NotImplementedError  # type: ignore[misc]

    def get_value(self) -> float:
        return self._values[-1] if self._values else 0.0


class KalmanFilter(AdaptiveFilter):
    """
    Kalman Filter for noise reduction and trend estimation.
    Adaptively adjusts to market changes.
    """

    def __init__(self, process_variance: float = 1e-5, measurement_variance: float = 1e-3):
        super().__init__("Kalman")
        
        self._process_variance = process_variance
        self._measurement_variance = measurement_variance
        
        self._estimate = 0.0
        self._error_estimate = 1.0
        self._error_measurement = 1.0
        self._kalman_gain = 0.0
        
        self._initialized = False

    def update(self, value: float) -> float:
        """Update filter with new observation."""
        
        if not self._initialized:
            self._estimate = value
            self._initialized = True
            self._values.append(value)
            return value
        
        self._kalman_gain = self._error_estimate / (self._error_estimate + self._measurement_variance)
        
        self._estimate = self._estimate + self._kalman_gain * (value - self._estimate)
        
        self._error_estimate = (1 - self._kalman_gain) * self._error_estimate + abs(self._estimate - value) * self._process_variance
        
        self._values.append(self._estimate)
        
        if len(self._values) > 100:
            self._values = self._values[-50:]
        
        return self._estimate

    def reset(self) -> None:
        """Reset filter state."""
        self._initialized = False
        self._estimate = 0.0
        self._error_estimate = 1.0
        self._values.clear()


class KAMAFilter(AdaptiveFilter):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Automatically adjusts to market volatility.
    """

    def __init__(self, fast_period: int = 2, slow_period: int = 30, period: int = 10):
        super().__init__("KAMA")
        
        self._fast_period = fast_period
        self._slow_period = slow_period
        self._period = period
        
        self._prices: list[float] = []
        self._kama_value = 0.0

    def update(self, value: float) -> float:
        """Update KAMA with new price."""
        
        self._prices.append(value)
        
        if len(self._prices) > self._period * 2:
            self._prices = self._prices[-(self._period * 2):]
        
        if len(self._prices) < self._period:
            self._kama_value = value
            self._values.append(self._kama_value)
            return self._kama_value
        
        direction = abs(value - self._prices[-(self._period)])
        
        volatility = 0.0
        for i in range(len(self._prices) - self._period):
            volatility += abs(self._prices[i + self._period] - self._prices[i])
        
        if volatility > 0:
            er = direction / volatility
        else:
            er = 0.0
        
        sc = (er * (2.0 / (self._fast_period + 1) - 2.0 / (self._slow_period + 1)) + 2.0 / (self._slow_period + 1)) ** 2
        
        if self._kama_value == 0:
            self._kama_value = value
        else:
            self._kama_value = self._kama_value + sc * (value - self._kama_value)
        
        self._values.append(self._kama_value)
        
        return self._kama_value

    def get_trend_direction(self) -> str:
        """Get current trend direction based on KAMA."""
        
        if len(self._values) < 10:
            return "NEUTRAL"
        
        recent = self._values[-10:]
        
        if recent[-1] > recent[0] * 1.01:
            return "BULLISH"
        elif recent[-1] < recent[0] * 0.99:
            return "BEARISH"
        
        return "SIDEWAYS"


class AdaptiveFilters:
    """
    Collection of adaptive filters for market analysis.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self.kalman = KalmanFilter()
        self.kama = KAMAFilter()
        
        self._price_history: list[float] = []

    def update(self, price: float) -> dict[str, float | str]:
        """Update all filters with new price."""
        
        self._price_history.append(price)
        
        if len(self._price_history) > 200:
            self._price_history = self._price_history[-100:]
        
        kalman_value = self.kalman.update(price)
        kama_value = self.kama.update(price)
        
        return {
            "kalman": kalman_value,
            "kama": kama_value,
            "trend": self.kama.get_trend_direction(),
        }

    def get_smoothed_price(self) -> float:
        """Get smoothed price estimate."""
        return self.kalman.get_value()

    def get_status(self) -> dict[str, Any]:
        """Get filter status."""
        return {
            "kalman": {
                "initialized": self.kalman._initialized,
                "current_value": self.kalman.get_value(),
            },
            "kama": {
                "current_value": self.kama.get_value(),
                "trend": self.kama.get_trend_direction(),
            },
            "price_history_count": len(self._price_history),
        }
