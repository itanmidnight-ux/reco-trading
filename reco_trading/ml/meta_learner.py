from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class Task:
    name: str
    train_data: np.ndarray
    train_labels: np.ndarray
    test_data: np.ndarray
    test_labels: np.ndarray
    metadata: dict = field(default_factory=dict)


@dataclass
class MetaParameters:
    meta_lr: float = 0.001
    inner_lr: float = 0.01
    inner_steps: int = 5
    outer_steps: int = 100
    support_size: int = 5
    query_size: int = 10
    hidden_size: int = 64


class MetaLearner:
    def __init__(self, config: MetaParameters | None = None):
        self.config = config or MetaParameters()
        self.logger = logging.getLogger(__name__)
        
        self._meta_weights: dict[str, np.ndarray] = {}
        self._task_history: list[dict] = []
        self._performance_cache: dict[str, list[float]] = {}
        
        self._initialize_weights()
        
        self.logger.info("Meta-Learner initialized")

    def _initialize_weights(self) -> None:
        self._meta_weights = {
            "input_weights": np.random.randn(20, 64) * 0.01,
            "hidden_weights": np.random.randn(64, 32) * 0.01,
            "output_weights": np.random.randn(32, 3) * 0.01,
            "input_bias": np.zeros(64),
            "hidden_bias": np.zeros(32),
            "output_bias": np.zeros(3)
        }

    def _forward(self, x: np.ndarray, weights: dict[str, np.ndarray] | None = None) -> np.ndarray:
        if weights is None:
            weights = self._meta_weights
        
        h1 = np.tanh(x @ weights["input_weights"] + weights["input_bias"])
        h2 = np.tanh(h1 @ weights["hidden_weights"] + weights["hidden_bias"])
        output = softmax(h2 @ weights["output_weights"] + weights["output_bias"])
        
        return output

    def _compute_loss(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        labels_idx = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
        ce = -np.log(predictions[np.arange(len(predictions)), labels_idx] + 1e-8)
        return np.mean(ce)

    def _gradient(self, x: np.ndarray, y: np.ndarray, weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Compute gradients for backpropagation - FIXED dimension alignment."""
        # Forward pass to get activations
        z1 = x @ weights["input_weights"] + weights["input_bias"]
        h1 = np.tanh(z1)
        
        z2 = h1 @ weights["hidden_weights"] + weights["hidden_bias"]
        h2 = np.tanh(z2)
        
        output = h2 @ weights["output_weights"] + weights["output_bias"]
        
        # Compute error at output
        error = output - y
        
        # Gradient for output layer
        grad_output = error / len(x)
        
        # Gradient for hidden layer - use h2 (hidden2 activation) NOT input
        tanh_deriv_h2 = (1 - h2 ** 2)
        grad_hidden = (grad_output @ weights["output_weights"].T) * tanh_deriv_h2
        
        # Gradient for input layer - use h1 (hidden1 activation)
        tanh_deriv_h1 = (1 - h1 ** 2)
        grad_input = (grad_hidden @ weights["hidden_weights"].T) * tanh_deriv_h1
        
        return {
            "output_weights": h2.T @ grad_output,
            "hidden_weights": h1.T @ grad_hidden,
            "input_weights": x.T @ grad_input,
            "output_bias": np.sum(grad_output, axis=0),
            "hidden_bias": np.sum(grad_hidden, axis=0),
            "input_bias": np.sum(grad_input, axis=0)
        }

    def _inner_update(self, support_x: np.ndarray, support_y: np.ndarray,
                     weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        adapted_weights = {k: v.copy() for k, v in weights.items()}
        
        for _ in range(self.config.inner_steps):
            grad = self._gradient(support_x, support_y, adapted_weights)
            
            for key in adapted_weights:
                adapted_weights[key] -= self.config.inner_lr * grad[key]
        
        return adapted_weights

    def _outer_update(self, support_x: np.ndarray, support_y: np.ndarray,
                     query_x: np.ndarray, query_y: np.ndarray) -> dict[str, np.ndarray]:
        adapted_weights = self._inner_update(support_x, support_y, self._meta_weights)
        
        grad = self._gradient(query_x, query_y, adapted_weights)
        
        for key in self._meta_weights:
            self._meta_weights[key] -= self.config.meta_lr * grad[key]
        
        return self._meta_weights.copy()

    def _adapt_to_task(self, task: Task) -> dict[str, np.ndarray]:
        support_size = min(self.config.support_size, len(task.train_data))
        indices = np.random.choice(len(task.train_data), support_size, replace=False)
        
        support_x = task.train_data[indices]
        support_y = task.train_labels[indices]
        
        return self._inner_update(support_x, support_y, self._meta_weights)

    def train_on_task(self, task: Task) -> dict[str, Any]:
        if len(task.train_data) < self.config.support_size + self.config.query_size:
            return {"error": "Insufficient data for meta-learning"}
        
        indices = np.random.permutation(len(task.train_data))
        support_idx = indices[:self.config.support_size]
        query_idx = indices[self.config.support_size:self.config.support_size + self.config.query_size]
        
        support_x = task.train_data[support_idx]
        support_y = task.train_labels[support_idx]
        query_x = task.train_data[query_idx]
        query_y = task.train_labels[query_idx]
        
        for step in range(self.config.outer_steps):
            self._outer_update(support_x, support_y, query_x, query_y)
            
            if step % 10 == 0:
                adapted = self._adapt_to_task(task)
                predictions = self._forward(task.test_data, adapted)
                loss = self._compute_loss(predictions, task.test_labels)
                self.logger.debug(f"Step {step}, Test loss: {loss:.4f}")
        
        adapted = self._adapt_to_task(task)
        final_predictions = self._forward(task.test_data, adapted)
        final_loss = self._compute_loss(final_predictions, task.test_labels)
        
        accuracy = np.mean(np.argmax(final_predictions, axis=1) == np.argmax(task.test_labels, axis=1))
        
        self._task_history.append({
            "task_name": task.name,
            "final_loss": final_loss,
            "accuracy": accuracy,
            "timestamp": datetime.now()
        })
        
        return {
            "task_name": task.name,
            "final_loss": float(final_loss),
            "accuracy": float(accuracy),
            "meta_weights": {k: v.shape for k, v in self._meta_weights.items()}
        }

    def predict(self, x: np.ndarray, adapted_weights: dict[str, np.ndarray] | None = None) -> np.ndarray:
        if adapted_weights is None:
            adapted_weights = self._meta_weights
        
        return self._forward(x, adapted_weights)

    def fast_adapt(self, task: Task, num_adapt_steps: int | None = None) -> dict[str, np.ndarray]:
        steps = num_adapt_steps or self.config.inner_steps
        
        support_size = min(self.config.support_size, len(task.train_data))
        indices = np.random.choice(len(task.train_data), support_size, replace=False)
        
        support_x = task.train_data[indices]
        support_y = task.train_labels[indices]
        
        adapted = {k: v.copy() for k, v in self._meta_weights.items()}
        
        for _ in range(steps):
            grad = self._gradient(support_x, support_y, adapted)
            for key in adapted:
                adapted[key] -= self.config.inner_lr * grad[key]
        
        return adapted

    def get_performance_history(self, task_name: str | None = None) -> list[dict]:
        if task_name:
            return [t for t in self._task_history if t["task_name"] == task_name]
        return self._task_history

    def get_meta_stats(self) -> dict:
        if not self._task_history:
            return {"total_tasks": 0}
        
        accuracies = [t["accuracy"] for t in self._task_history]
        losses = [t["final_loss"] for t in self._task_history]
        
        return {
            "total_tasks": len(self._task_history),
            "avg_accuracy": np.mean(accuracies),
            "avg_loss": np.mean(losses),
            "best_accuracy": max(accuracies),
            "recent_tasks": [t["task_name"] for t in self._task_history[-5:]]
        }


class MarketMetaLearner:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self._meta_learner = MetaLearner()
        self._regime_adapters: dict[str, dict[str, np.ndarray]] = {}
        self._pair_adapters: dict[str, dict[str, np.ndarray]] = {}
        
        self._default_task = Task(
            name="default",
            train_data=np.random.randn(100, 20),
            train_labels=np.random.randint(0, 3, 100),
            test_data=np.random.randn(20, 20),
            test_labels=np.random.randint(0, 3, 20)
        )

    def create_market_task(self, symbol: str, features: np.ndarray, 
                           labels: np.ndarray) -> Task:
        return Task(
            name=symbol,
            train_data=features[:int(len(features) * 0.8)],
            train_labels=labels[:int(len(labels) * 0.8)],
            test_data=features[int(len(features) * 0.8):],
            test_labels=labels[int(len(labels) * 0.8):]
        )

    def adapt_to_market(self, symbol: str, regime: str, features: np.ndarray,
                       labels: np.ndarray) -> dict:
        task = self.create_market_task(symbol, features, labels)
        
        result = self._meta_learner.train_on_task(task)
        
        adapted_weights = self._meta_learner.fast_adapt(task)
        
        if regime not in self._regime_adapters:
            self._regime_adapters[regime] = {}
        self._regime_adapters[regime] = adapted_weights
        
        self._pair_adapters[symbol] = adapted_weights
        
        return result

    def get_adapted_weights(self, symbol: str | None = None, 
                           regime: str | None = None) -> dict[str, np.ndarray] | None:
        if symbol and symbol in self._pair_adapters:
            return self._pair_adapters[symbol]
        
        if regime and regime in self._regime_adapters:
            return self._regime_adapters[regime]
        
        return None

    def predict_with_adaptation(self, symbol: str, regime: str,
                               features: np.ndarray) -> dict:
        adapted = self.get_adapted_weights(symbol, regime)
        
        if adapted is None:
            adapted = self._meta_learner.fast_adapt(
                self._default_task,
                num_adapt_steps=3
            )
        
        predictions = self._meta_learner.predict(features, adapted)
        
        action = np.argmax(predictions, axis=1)
        confidence = np.max(predictions, axis=1)
        
        return {
            "action": ["SELL", "HOLD", "BUY"][action[0]] if len(action) > 0 else "HOLD",
            "confidence": float(confidence[0]) if len(confidence) > 0 else 0.33,
            "probabilities": predictions[0].tolist() if len(predictions) > 0 else [0.33, 0.34, 0.33]
        }

    def get_meta_stats(self) -> dict:
        return {
            "meta_learner": self._meta_learner.get_meta_stats(),
            "regimes_adapted": list(self._regime_adapters.keys()),
            "pairs_adapted": list(self._pair_adapters.keys())
        }


class QuickAdapter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._quick_weights: dict[str, np.ndarray] = {}
        self._initialize_quick_weights()

    def _initialize_quick_weights(self) -> None:
        self._quick_weights = {
            "W1": np.random.randn(20, 32) * 0.1,
            "b1": np.zeros(32),
            "W2": np.random.randn(32, 3) * 0.1,
            "b2": np.zeros(3)
        }

    def quick_adapt(self, examples: np.ndarray, labels: np.ndarray,
                   steps: int = 10, lr: float = 0.1) -> dict[str, np.ndarray]:
        adapted = {k: v.copy() for k, v in self._quick_weights.items()}
        
        for _ in range(steps):
            h = np.tanh(examples @ adapted["W1"] + adapted["b1"])
            out = softmax(h @ adapted["W2"] + adapted["b2"])
            
            error = out - labels
            grad_out = error / len(examples)
            grad_h = grad_out @ adapted["W2"].T * (1 - h ** 2)
            
            adapted["W2"] -= lr * (h.T @ grad_out)
            adapted["b2"] -= lr * np.sum(grad_out, axis=0)
            adapted["W1"] -= lr * (examples.T @ grad_h)
            adapted["b1"] -= lr * np.sum(grad_h, axis=0)
        
        return adapted

    def predict(self, x: np.ndarray, weights: dict[str, np.ndarray] | None = None) -> np.ndarray:
        w = weights or self._quick_weights
        h = np.tanh(x @ w["W1"] + w["b1"])
        return softmax(h @ w["W2"] + w["b2"])


def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def create_few_shot_task(features: np.ndarray, labels: np.ndarray,
                        n_support: int = 5, n_query: int = 10) -> Task:
    indices = np.random.permutation(len(features))
    
    return Task(
        name="few_shot_task",
        train_data=features[indices[:n_support + n_query]],
        train_labels=labels[indices[:n_support + n_query]],
        test_data=features[indices[n_support + n_query:]],
        test_labels=labels[indices[n_support + n_query:]],
        metadata={"n_support": n_support, "n_query": n_query}
    )