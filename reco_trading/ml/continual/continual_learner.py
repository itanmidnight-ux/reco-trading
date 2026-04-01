from __future__ import annotations

import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    timestamp: datetime = field(default_factory=datetime.now)
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class KnowledgeSnapshot:
    model_weights: dict
    performance_metrics: dict
    training_steps: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PlasticityConfig:
    max_buffer_size: int = 10000
    replay_batch_size: int = 32
    learning_rate: float = 0.001
    discount_factor: float = 0.95
    plasticity_decay: float = 0.95
    consolidation_threshold: float = 0.85
    min_samples_for_update: int = 50


class ContinualLearner:
    def __init__(self, config: PlasticityConfig | None = None):
        self.config = config or PlasticityConfig()
        self.logger = logging.getLogger(__name__)
        
        self._experience_buffer: deque[Experience] = deque(maxlen=self.config.max_buffer_size)
        self._knowledge_snapshots: deque[KnowledgeSnapshot] = deque(maxlen=10)
        
        self._model_weights: dict[str, np.ndarray] = {}
        self._initialize_model()
        
        self._training_step = 0
        self._plasticity_score = 1.0
        
        self._performance_history: list[dict] = []
        
        self.logger.info("ContinualLearner initialized")

    def _initialize_model(self) -> None:
        np.random.seed(42)
        
        self._model_weights = {
            "W1": np.random.randn(20, 64) * 0.1,
            "b1": np.zeros(64),
            "W2": np.random.randn(64, 32) * 0.1,
            "b2": np.zeros(32),
            "W3": np.random.randn(32, 3) * 0.1,
            "b3": np.zeros(3)
        }

    def add_experience(self, state: np.ndarray, action: int, reward: float,
                      next_state: np.ndarray, done: bool) -> None:
        exp = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        )
        
        self._experience_buffer.append(exp)
        
        if len(self._experience_buffer) >= self.config.min_samples_for_update:
            self._update_plasticity_score()

    def _update_plasticity_score(self) -> None:
        recent_rewards = [e.reward for e in list(self._experience_buffer)[-100:]]
        
        if not recent_rewards:
            return
        
        avg_reward = np.mean(recent_rewards)
        reward_variance = np.std(recent_rewards)
        
        stability = 1.0 / (1.0 + reward_variance)
        
        if avg_reward > 0:
            self._plasticity_score = min(
                1.0,
                self._plasticity_score * self.config.plasticity_decay + 
                (1 - self.config.plasticity_decay) * stability
            )
        else:
            self._plasticity_score = max(
                0.3,
                self._plasticity_score * self.config.plasticity_decay
            )

    def sample_batch(self) -> list[Experience]:
        if len(self._experience_buffer) < self.config.replay_batch_size:
            return list(self._experience_buffer)
        
        return np.random.choice(
            list(self._experience_buffer),
            self.config.replay_batch_size,
            replace=False
        ).tolist()

    def _forward(self, x: np.ndarray) -> np.ndarray:
        h1 = np.tanh(x @ self._model_weights["W1"] + self._model_weights["b1"])
        h2 = np.tanh(h1 @ self._model_weights["W2"] + self._model_weights["b2"])
        output = softmax(h2 @ self._model_weights["W3"] + self._model_weights["b3"])
        return output

    def _compute_gradient(self, batch: list[Experience]) -> dict[str, np.ndarray]:
        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch])
        
        h1 = np.tanh(states @ self._model_weights["W1"] + self._model_weights["b1"])
        h2 = np.tanh(h1 @ self._model_weights["W2"] + self._model_weights["b2"])
        current_q = softmax(h2 @ self._model_weights["W3"] + self._model_weights["b3"])
        next_q = self._forward(next_states)
        
        target_q = rewards + self.config.discount_factor * np.max(next_q, axis=1) * (1 - dones)
        
        current_q_action = current_q[np.arange(len(actions)), actions]
        
        error = current_q_action - target_q
        grad_output = np.zeros_like(current_q)
        grad_output[np.arange(len(actions)), actions] = error
        
        grad_h2 = grad_output @ self._model_weights["W3"].T * (1 - h2 ** 2)
        
        grad_W3 = h2.T @ grad_output
        grad_b3 = np.sum(grad_output, axis=0)
        
        grad_h1 = grad_h2 @ self._model_weights["W2"].T * (1 - h1 ** 2)
        
        grad_W2 = h1.T @ grad_h2
        grad_b2 = np.sum(grad_h2, axis=0)
        
        grad_W1 = states.T @ grad_h1
        grad_b1 = np.sum(grad_h1, axis=0)
        
        return {
            "W1": grad_W1 / len(batch),
            "b1": grad_b1 / len(batch),
            "W2": grad_W2 / len(batch),
            "b2": grad_b2 / len(batch),
            "W3": grad_W3 / len(batch),
            "b3": grad_b3 / len(batch)
        }

    def train_step(self) -> dict[str, float]:
        if len(self._experience_buffer) < self.config.min_samples_for_update:
            return {"status": "insufficient_data"}
        
        batch = self.sample_batch()
        
        if self._plasticity_score < self.config.consolidation_threshold:
            self.logger.info(f"Low plasticity ({self._plasticity_score:.2f}), using knowledge consolidation")
            return self._consolidate_knowledge()
        
        gradients = self._compute_gradient(batch)
        
        lr = self.config.learning_rate * self._plasticity_score
        
        for key in self._model_weights:
            self._model_weights[key] -= lr * gradients.get(key, np.zeros_like(self._model_weights[key]))
        
        self._training_step += 1
        
        avg_reward = np.mean([e.reward for e in batch])
        
        return {
            "status": "updated",
            "training_step": self._training_step,
            "avg_reward": float(avg_reward),
            "plasticity_score": float(self._plasticity_score),
            "buffer_size": len(self._experience_buffer)
        }

    def _consolidate_knowledge(self) -> dict[str, float]:
        snapshot = KnowledgeSnapshot(
            model_weights={k: v.copy() for k, v in self._model_weights.items()},
            performance_metrics=self.get_performance_metrics(),
            training_steps=self._training_step
        )
        
        self._knowledge_snapshots.append(snapshot)
        
        self._plasticity_score = min(1.0, self._plasticity_score + 0.1)
        
        return {
            "status": "consolidated",
            "plasticity_score": float(self._plasticity_score),
            "snapshots_stored": len(self._knowledge_snapshots)
        }

    def predict(self, state: np.ndarray) -> np.ndarray:
        return self._forward(state.reshape(1, -1))

    def get_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        if np.random.rand() < epsilon:
            return np.random.randint(0, 3)
        
        q_values = self.predict(state)[0]
        return int(np.argmax(q_values))

    def get_performance_metrics(self) -> dict:
        if not self._experience_buffer:
            return {"total_experiences": 0}
        
        recent = list(self._experience_buffer)[-100:]
        
        avg_reward = np.mean([e.reward for e in recent])
        positive_rate = sum(1 for e in recent if e.reward > 0) / len(recent)
        
        return {
            "total_experiences": len(self._experience_buffer),
            "avg_reward_100": float(avg_reward),
            "positive_rate": float(positive_rate),
            "plasticity_score": float(self._plasticity_score),
            "training_steps": self._training_step,
            "snapshots": len(self._knowledge_snapshots)
        }

    def get_knowledge_distillation(self, target_state: np.ndarray) -> np.ndarray:
        distilled = np.zeros(3)
        
        for snapshot in self._knowledge_snapshots:
            q_values = self._forward_with_weights(target_state.reshape(1, -1), snapshot.model_weights)
            distilled += q_values[0]
        
        if self._knowledge_snapshots:
            distilled /= len(self._knowledge_snapshots)
        
        return distilled

    def _forward_with_weights(self, x: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
        h1 = np.tanh(x @ weights["W1"] + weights["b1"])
        h2 = np.tanh(h1 @ weights["W2"] + weights["b2"])
        return softmax(h2 @ weights["W3"] + weights["b3"])


class ExperienceBuffer:
    def __init__(self, max_size: int = 10000):
        self.buffer: deque[Experience] = deque(maxlen=max_size)
        self._episode_buffers: dict[str, list[Experience]] = {}

    def add(self, experience: Experience) -> None:
        self.buffer.append(experience)
        
        if experience.episode_id not in self._episode_buffers:
            self._episode_buffers[experience.episode_id] = []
        self._episode_buffers[experience.episode_id].append(experience)

    def sample(self, batch_size: int) -> list[Experience]:
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        return list(np.random.choice(list(self.buffer), batch_size, replace=False))

    def get_episode(self, episode_id: str) -> list[Experience]:
        return self._episode_buffers.get(episode_id, [])

    def get_all_episodes(self) -> list[list[Experience]]:
        return list(self._episode_buffers.values())

    def clear(self) -> None:
        self.buffer.clear()
        self._episode_buffers.clear()


class OnlineTrainer:
    def __init__(self, learner: ContinualLearner, update_interval: int = 10):
        self.learner = learner
        self.update_interval = update_interval
        self._update_counter = 0
        self._is_running = False

    async def start(self) -> None:
        self._is_running = True
        self.logger.info("OnlineTrainer started")

    async def stop(self) -> None:
        self._is_running = False
        self.logger.info("OnlineTrainer stopped")

    def should_update(self) -> bool:
        self._update_counter += 1
        return self._update_counter >= self.update_interval

    def update(self) -> dict:
        if not self.should_update():
            return {"status": "waiting", "counter": self._update_counter}
        
        self._update_counter = 0
        return self.learner.train_step()

    def add_experience(self, state: np.ndarray, action: int, reward: float,
                      next_state: np.ndarray, done: bool) -> None:
        self.learner.add_experience(state, action, reward, next_state, done)


def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def extract_features(state_dict: dict) -> np.ndarray:
    features = []
    
    for key in ["price", "volume", "rsi", "macd", "atr", "volatility"]:
        if key in state_dict:
            features.append(float(state_dict[key]))
        else:
            features.append(0.0)
    
    while len(features) < 20:
        features.append(0.0)
    
    return np.array(features[:20])