from __future__ import annotations

import logging
import math
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class QLearningAgent:
    """
    Q-Learning Reinforcement Learning Agent for trading decisions.
    Learns optimal actions based on market states.
    """

    def __init__(
        self,
        state_size: int = 20,
        action_size: int = 3,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        self.logger = logging.getLogger(__name__)
        
        self.state_size = state_size
        self.action_size = action_size  # 0: SELL, 1: HOLD, 2: BUY
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: state -> action values
        self.q_table: dict[tuple, np.ndarray] = {}
        
        # Experience replay buffer
        self.memory: deque = deque(maxlen=10000)
        
        # Training history
        self.episode_count = 0
        self.total_reward = 0.0
        
    def get_state_key(self, features: list[float]) -> tuple:
        """Convert continuous features to discrete state key"""
        # Discretize features into bins
        bins = 10
        discretized = []
        for f in features[:self.state_size]:
            bin_idx = min(int(f * bins), bins - 1) if f >= 0 else max(int(f * bins), -bins + 1)
            discretized.append(bin_idx)
        return tuple(discretized)
    
    def get_q_values(self, state: tuple) -> np.ndarray:
        """Get Q-values for a state"""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)
        return self.q_table[state]
    
    def choose_action(self, state: tuple, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        q_values = self.get_q_values(state)
        return int(np.argmax(q_values))
    
    def learn(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        done: bool,
    ) -> None:
        """Update Q-values using Q-learning"""
        # Store in memory
        self.memory.append((state, action, reward, next_state, done))
        
        # Get current Q-value
        current_q = self.get_q_values(state)[action]
        
        # Get max Q-value for next state
        next_q_max = np.max(self.get_q_values(next_state)) if not done else 0
        
        # Q-learning update
        target_q = reward + self.discount_factor * next_q_max
        new_q = current_q + self.learning_rate * (target_q - current_q)
        
        # Update Q-table
        q_values = self.get_q_values(state)
        q_values[action] = new_q
        self.q_table[state] = q_values
        
        self.total_reward += reward
        
        # Decay epsilon
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.episode_count += 1
    
    def extract_features(self, market_data: dict[str, Any]) -> list[float]:
        """Extract features from market data for state representation"""
        features = []
        
        # Price features
        price = market_data.get("price", 0)
        prev_price = market_data.get("prev_price", price)
        price_change = (price - prev_price) / prev_price if prev_price else 0
        features.append(price_change)
        
        # Volume features
        volume = market_data.get("volume", 0)
        avg_volume = market_data.get("avg_volume", volume)
        volume_ratio = volume / avg_volume if avg_volume else 1
        features.append(volume_ratio - 1)  # Normalize around 0
        
        # Volatility
        volatility = market_data.get("volatility", 0)
        features.append(volatility * 10)  # Scale up
        
        # RSI
        rsi = market_data.get("rsi", 50)
        features.append((rsi - 50) / 50)  # Normalize to -1 to 1
        
        # MACD
        macd = market_data.get("macd", 0)
        features.append(macd)
        
        # Trend indicators
        sma_20 = market_data.get("sma_20", price)
        sma_50 = market_data.get("sma_50", price)
        trend = (sma_20 - sma_50) / sma_50 if sma_50 else 0
        features.append(trend)
        
        # Bollinger Bands position
        bb_position = market_data.get("bb_position", 0.5)
        features.append(bb_position - 0.5)
        
        # Additional features (pad to state_size)
        while len(features) < self.state_size:
            features.append(0.0)
        
        return features[:self.state_size]
    
    def get_status(self) -> dict[str, Any]:
        """Get agent status"""
        return {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "epsilon": self.epsilon,
            "episode_count": self.episode_count,
            "total_reward": self.total_reward,
            "q_table_size": len(self.q_table),
            "memory_size": len(self.memory),
        }


class PolicyGradientAgent:
    """
    Policy Gradient Agent (REINFORCE) for trading.
    Learns probability distributions over actions.
    """

    def __init__(
        self,
        state_size: int = 20,
        action_size: int = 3,
        learning_rate: float = 0.001,
    ):
        self.logger = logging.getLogger(__name__)
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Policy network weights (simple linear)
        self.weights = np.random.randn(state_size, action_size) * 0.01
        
        # Memory for episodes
        self.episode_states: list[np.ndarray] = []
        self.episode_actions: list[int] = []
        self.episode_rewards: list[float] = []
        
    def forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass - compute action probabilities"""
        logits = np.dot(state, self.weights)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return probs
    
    def choose_action(self, state: np.ndarray) -> int:
        """Sample action from policy"""
        probs = self.forward(state)
        return np.random.choice(self.action_size, p=probs)
    
    def store_transition(self, state: np.ndarray, action: int, reward: float) -> None:
        """Store transition for episode"""
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
    
    def learn(self) -> float:
        """Update policy using REINFORCE"""
        if not self.episode_states:
            return 0.0
        
        # Compute discounted rewards
        rewards = np.array(self.episode_rewards)
        discounts = np.power(0.99, np.arange(len(rewards)))
        discounted = rewards * discounts
        
        # Normalize rewards
        if np.std(discounted) > 0:
            discounted = (discounted - np.mean(discounted)) / np.std(discounted)
        
        # Policy gradient update
        policy_loss = 0.0
        for i, (state, action, reward) in enumerate(zip(
            self.episode_states, self.episode_actions, discounted
        )):
            probs = self.forward(state)
            log_prob = np.log(probs[action] + 1e-10)
            policy_loss -= log_prob * reward
        
        # Gradient update
        if len(self.episode_states) > 0:
            gradient = policy_loss / len(self.episode_states)
            self.weights -= self.learning_rate * gradient
        
        # Clear episode memory
        total_reward = sum(self.episode_rewards)
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        
        return total_reward
    
    def get_status(self) -> dict[str, Any]:
        return {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "learning_rate": self.learning_rate,
            "episode_length": len(self.episode_states),
        }


class EnsemblePredictor:
    """
    Ensemble of multiple models for robust predictions.
    Combines: Bayesian, GA, RL, and Technical indicators.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.rl_agent = QLearningAgent(
            state_size=20,
            action_size=3,
            learning_rate=0.1,
            discount_factor=0.95,
        )
        
        self.policy_agent = PolicyGradientAgent(
            state_size=20,
            action_size=3,
            learning_rate=0.001,
        )
        
        # Model weights (learned)
        self.model_weights = {
            "rl": 0.3,
            "policy": 0.2,
            "technical": 0.3,
            "bayesian": 0.2,
        }
        
        # Historical predictions
        self.prediction_history: list[dict] = []
        
    def predict(
        self,
        market_data: dict[str, Any],
        bayesian_score: float = 0.5,
    ) -> dict[str, Any]:
        """Ensemble prediction from multiple models"""
        
        # Extract features
        features = self.rl_agent.extract_features(market_data)
        state = self.rl_agent.get_state_key(features)
        
        # Get RL prediction
        rl_action = self.rl_agent.choose_action(state, training=False)
        rl_probs = self.rl_agent.get_q_values(state)
        
        # Get Policy Gradient prediction
        policy_action = self.policy_agent.choose_action(np.array(features))
        
        # Technical indicators prediction
        technical_signal = self._technical_prediction(market_data)
        
        # Bayesian input
        bayesian_signal = 2 if bayesian_score > 0.7 else (0 if bayesian_score < 0.3 else 1)
        
        # Weighted ensemble
        signal_votes = {0: 0.0, 1: 0.0, 2: 0.0}  # SELL, HOLD, BUY
        
        # RL vote
        signal_votes[rl_action] += self.model_weights["rl"]
        
        # Policy vote
        signal_votes[policy_action] += self.model_weights["policy"]
        
        # Technical vote
        signal_votes[technical_signal] += self.model_weights["technical"]
        
        # Bayesian vote
        signal_votes[bayesian_signal] += self.model_weights["bayesian"]
        
        # Final decision
        final_action = max(signal_votes, key=signal_votes.get)
        confidence = signal_votes[final_action]
        
        # Map to trading signal
        signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        
        result = {
            "signal": signal_map[final_action],
            "confidence": min(confidence, 1.0),
            "votes": signal_votes,
            "rl_action": signal_map[rl_action],
            "policy_action": signal_map[policy_action],
            "technical_action": signal_map[technical_signal],
            "bayesian_action": signal_map[bayesian_signal],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        self.prediction_history.append(result)
        if len(self.prediction_history) > 100:
            self.prediction_history = self.prediction_history[-50:]
        
        return result
    
    def _technical_prediction(self, market_data: dict[str, Any]) -> int:
        """Technical indicators based prediction"""
        score = 0
        
        # RSI
        rsi = market_data.get("rsi", 50)
        if rsi < 30:
            score += 1  # Oversold - BUY
        elif rsi > 70:
            score -= 1  # Overbought - SELL
        
        # MACD
        macd = market_data.get("macd", 0)
        if macd > 0:
            score += 0.5
        else:
            score -= 0.5
        
        # Trend
        price = market_data.get("price", 0)
        sma_20 = market_data.get("sma_20", price)
        sma_50 = market_data.get("sma_50", price)
        
        if sma_20 > sma_50:
            score += 0.5
        else:
            score -= 0.5
        
        # Bollinger Bands
        bb_position = market_data.get("bb_position", 0.5)
        if bb_position < 0.2:
            score += 0.5
        elif bb_position > 0.8:
            score -= 0.5
        
        # Return action
        if score > 0.5:
            return 2  # BUY
        elif score < -0.5:
            return 0  # SELL
        else:
            return 1  # HOLD
    
    def update_weights(self, performance: dict[str, float]) -> None:
        """Update ensemble weights based on performance"""
        total = sum(performance.values())
        
        for key in self.model_weights:
            if key in performance:
                self.model_weights[key] = performance[key] / total
        
        self.logger.info(f"Updated ensemble weights: {self.model_weights}")
    
    def learn_from_trade(self, trade_result: dict[str, Any]) -> None:
        """Update RL agents based on trade result"""
        market_data = trade_result.get("market_data", {})
        
        # Extract features
        features = self.rl_agent.extract_features(market_data)
        state = self.rl_agent.get_state_key(features)
        
        # Determine reward
        pnl = trade_result.get("pnl", 0)
        if pnl > 0:
            reward = 1.0
        elif pnl < 0:
            reward = -1.0
        else:
            reward = 0.0
        
        # Get next state (simplified - same state for now)
        next_state = state
        
        # Determine action taken
        action = trade_result.get("action", 1)  # Default HOLD
        
        # Update RL agent
        self.rl_agent.learn(state, action, reward, next_state, done=True)
        
        # Update policy agent
        self.policy_agent.store_transition(np.array(features), action, reward)
        self.policy_agent.learn()
    
    def get_status(self) -> dict[str, Any]:
        return {
            "rl_agent": self.rl_agent.get_status(),
            "policy_agent": self.policy_agent.get_status(),
            "model_weights": self.model_weights,
            "prediction_count": len(self.prediction_history),
        }
