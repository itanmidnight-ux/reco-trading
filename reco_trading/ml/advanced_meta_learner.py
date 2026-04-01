import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import functional_call
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


@dataclass
class MetaLearningConfig:
    """Configuration for advanced meta-learning framework."""
    maml_inner_lr: float = 0.01
    maml_outer_lr: float = 0.001
    adaptation_steps: int = 5
    task_batch_size: int = 4
    max_tasks: int = 100
    adaptation_patience: int = 3
    memory_size: int = 1000
    confidence_threshold: float = 0.7


class MemoryAugmentedNetwork(nn.Module):
    """Memory-augmented neural network for meta-learning."""
    
    def __init__(self, input_size: int, hidden_size: int, memory_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        
        # Key and value networks for memory
        self.key_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.value_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Network for processing
        self.processing_network = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Output network
        self.output_network = nn.Linear(hidden_size, 3)  # BUY, SELL, HOLD
        
        # Memory matrix
        memory_keys = torch.randn(memory_size, hidden_size) * 0.1
        memory_values = torch.randn(memory_size, hidden_size) * 0.1
        
        self.register_buffer('memory_keys', memory_keys)
        self.register_buffer('memory_values', memory_values)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Compute query keys
        query_keys = self.key_network(x)
        
        # Compute attention over memory
        similarities = torch.matmul(query_keys, self.memory_keys.t())
        attention_weights = torch.softmax(similarities, dim=-1)
        
        # Retrieve memory values
        retrieved_values = torch.matmul(attention_weights, self.memory_values)
        
        # Process input with retrieved information
        combined_input = torch.cat([x, retrieved_values], dim=-1)
        processed = self.processing_network(combined_input)
        
        # Generate output
        output = self.output_network(processed)
        
        return output
    
    def write_to_memory(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Write new key-value pairs to memory."""
        batch_size = keys.size(0)
        
        # Simple FIFO replacement strategy
        for i in range(batch_size):
            idx = i % self.memory_size
            self.memory_keys[idx] = keys[i]
            self.memory_values[idx] = values[i]
    
    def get_memory_state(self) -> Dict[str, torch.Tensor]:
        """Get current memory state."""
        return {
            "keys": self.memory_keys.cpu().numpy(),
            "values": self.memory_values.cpu().numpy()
        }


class AdvancedMetaLearner(nn.Module):
    """Advanced meta-learning network with memory augmentation."""
    
    def __init__(self, config: MetaLearningConfig, input_size: int = 20):
        super().__init__()
        self.config = config
        self.input_size = input_size
        
        # Memory-augmented network
        self.memory_network = MemoryAugmentedNetwork(
            input_size=input_size,
            hidden_size=64,
            memory_size=config.memory_size
        )
        
        # Adaptation metrics
        self.adaptation_history: List[Dict[str, Any]] = []
        self.task_performance: Dict[str, List[float]] = {}
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.memory_network(x)
    
    def adapt_to_task(
        self, 
        support_data: torch.Tensor, 
        support_labels: torch.Tensor,
        num_steps: Optional[int] = None
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Adapt model to a new task using MAML-style adaptation."""
        steps = num_steps or self.config.adaptation_steps
        
        # Clone model for adaptation
        adapted_model = MemoryAugmentedNetwork(
            input_size=self.input_size,
            hidden_size=64,
            memory_size=self.config.memory_size
        )
        adapted_model.load_state_dict(self.memory_network.state_dict())
        
        optimizer = optim.Adam(adapted_model.parameters(), lr=self.config.maml_inner_lr)
        criterion = nn.CrossEntropyLoss()
        
        adaptation_history = []
        
        for step in range(steps):
            optimizer.zero_grad()
            
            # Forward pass
            predictions = adapted_model(support_data)
            loss = criterion(predictions, support_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track adaptation progress
            if step % 1 == 0:
                with torch.no_grad():
                    accuracy = torch.mean(
                        (predictions.argmax(dim=1) == support_labels).float()
                    ).item()
                    
                adaptation_history.append({
                    "step": step,
                    "loss": loss.item(),
                    "accuracy": accuracy
                })
        
        # Extract memory keys and values from support set
        with torch.no_grad():
            memory_keys = adapted_model.key_network(support_data)
            memory_values = adapted_model.value_network(support_data)
            
            # Write to memory
            adapted_model.write_to_memory(memory_keys, memory_values)
        
        # Calculate adaptation confidence
        final_accuracy = adaptation_history[-1]["accuracy"] if adaptation_history else 0
        adaptation_confidence = min(1.0, final_accuracy / self.config.confidence_threshold)
        
        return adapted_model, {
            "adaptation_history": adaptation_history,
            "final_accuracy": final_accuracy,
            "confidence": adaptation_confidence
        }
    
    def meta_update(
        self, 
        task_batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Dict[str, float]:
        """Perform meta-level update across multiple tasks."""
        meta_optimizer = optim.Adam(self.parameters(), lr=self.config.maml_outer_lr)
        criterion = nn.CrossEntropyLoss()
        
        meta_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        meta_accuracy = 0.0
        adaptation_metrics = []
        
        for support_x, support_y, query_x, query_y in task_batch:
            # Adapt to task using functional MAML (preserves gradient graph)
            adapted_params, adapt_info = self._functional_adapt(support_x, support_y)
            adaptation_metrics.append(adapt_info)
            
            # Evaluate on query set with gradients flowing back through adaptation
            query_predictions = functional_call(self.memory_network, adapted_params, query_x)
            query_loss = criterion(query_predictions, query_y)
            
            with torch.no_grad():
                query_accuracy = torch.mean(
                    (query_predictions.argmax(dim=1) == query_y).float()
                ).item()
            
            meta_loss = meta_loss + query_loss
            meta_accuracy += query_accuracy
        
        # Meta-update
        meta_loss = meta_loss / len(task_batch)
        meta_accuracy /= len(task_batch)
        
        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()
        
        # Store adaptation metrics
        for i, metrics in enumerate(adaptation_metrics):
            task_name = f"task_{i}"
            if task_name not in self.task_performance:
                self.task_performance[task_name] = []
            self.task_performance[task_name].append(metrics["final_accuracy"])
        
        return {
            "meta_loss": meta_loss.item(),
            "meta_accuracy": meta_accuracy,
            "avg_adaptation_accuracy": np.mean([m["final_accuracy"] for m in adaptation_metrics])
        }
    
    def _functional_adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Functional MAML-style adaptation that preserves gradient graph."""
        steps = self.config.adaptation_steps
        params = {k: v.clone() for k, v in self.memory_network.state_dict().items()}
        criterion = nn.CrossEntropyLoss()
        adaptation_history = []
        
        for step in range(steps):
            params = {k: v.clone().requires_grad_(True) for k, v in params.items()}
            predictions = functional_call(self.memory_network, params, support_x)
            loss = criterion(predictions, support_y)
            grads = torch.autograd.grad(loss, params.values(), create_graph=True)
            params = {k: v - self.config.maml_inner_lr * g for (k, v), g in zip(params.items(), grads)}
            
            if step % 1 == 0:
                with torch.no_grad():
                    accuracy = torch.mean(
                        (predictions.argmax(dim=1) == support_y).float()
                    ).item()
                adaptation_history.append({
                    "step": step,
                    "loss": loss.item(),
                    "accuracy": accuracy
                })
        
        final_accuracy = adaptation_history[-1]["accuracy"] if adaptation_history else 0
        adaptation_confidence = min(1.0, final_accuracy / self.config.confidence_threshold)
        
        return params, {
            "adaptation_history": adaptation_history,
            "final_accuracy": final_accuracy,
            "confidence": adaptation_confidence
        }


class MetaLearningManager:
    """Manager for meta-learning models and rapid market adaptation."""
    
    def __init__(self, config: Optional[MetaLearningConfig] = None):
        self.config = config or MetaLearningConfig()
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize meta-learner
        self.meta_learner = AdvancedMetaLearner(self.config).to(self.device)
        
        # Task storage
        self.market_tasks: Dict[str, Dict[str, Any]] = {}
        self.adapted_models: Dict[str, Tuple[nn.Module, Dict[str, Any]]] = {}
        
        # Performance tracking
        self.adaptation_history: List[Dict[str, Any]] = []
        
    def create_market_task(
        self, 
        pair: str, 
        features: np.ndarray, 
        labels: np.ndarray,
        regime: Optional[str] = None
    ) -> str:
        """Create a meta-learning task from market data."""
        
        task_name = f"{pair}_{regime or 'default'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Split data for support/query sets
        total_samples = len(features)
        support_size = min(20, total_samples // 2)
        query_size = min(20, total_samples - support_size)
        
        indices = np.random.permutation(total_samples)
        support_indices = indices[:support_size]
        query_indices = indices[support_size:support_size + query_size]
        
        # Store task
        self.market_tasks[task_name] = {
            "pair": pair,
            "regime": regime,
            "support_data": torch.FloatTensor(features[support_indices]).to(self.device),
            "support_labels": torch.LongTensor(labels[support_indices]).to(self.device),
            "query_data": torch.FloatTensor(features[query_indices]).to(self.device),
            "query_labels": torch.LongTensor(labels[query_indices]).to(self.device),
            "created_at": datetime.now(timezone.utc)
        }
        
        self.logger.info(f"Created meta-learning task: {task_name}")
        return task_name
    
    def adapt_to_market_condition(
        self, 
        pair: str, 
        regime: str,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """Adapt model to specific market condition."""
        
        try:
            # Create task from current market data
            task_name = self.create_market_task(pair, features, labels, regime)
            task = self.market_tasks[task_name]
            
            # Adapt to task
            adapted_model, adapt_info = self.meta_learner.adapt_to_task(
                task["support_data"],
                task["support_labels"]
            )
            
            # Store adapted model
            model_key = f"{pair}_{regime}"
            self.adapted_models[model_key] = (adapted_model, adapt_info)
            
            # Record adaptation
            adaptation_record = {
                "pair": pair,
                "regime": regime,
                "timestamp": datetime.now(timezone.utc),
                "task_name": task_name,
                "adaptation_info": adapt_info
            }
            self.adaptation_history.append(adaptation_record)
            
            self.logger.info(
                f"Adapted to {pair} in {regime} regime - "
                f"Accuracy: {adapt_info['final_accuracy']:.2%} - "
                f"Confidence: {adapt_info['confidence']:.2%}"
            )
            
            return {
                "pair": pair,
                "regime": regime,
                "task_name": task_name,
                "adaptation_successful": True,
                **adapt_info
            }
            
        except Exception as exc:
            self.logger.error(f"Market adaptation failed for {pair}: {exc}")
            return {"pair": pair, "regime": regime, "adaptation_successful": False, "error": str(exc)}
    
    def predict_with_adaptation(
        self, 
        pair: str, 
        regime: str, 
        features: np.ndarray
    ) -> Dict[str, Any]:
        """Make predictions using adapted model if available."""
        
        model_key = f"{pair}_{regime}"
        
        if model_key not in self.adapted_models:
            # Try base model prediction
            try:
                with torch.no_grad():
                    x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                    predictions = self.meta_learner(x)
                    probabilities = torch.softmax(predictions, dim=1).cpu().numpy()[0]
                    
                    action = np.argmax(probabilities)
                    confidence = np.max(probabilities)
                    
                    return {
                        "pair": pair,
                        "regime": regime,
                        "action": ["SELL", "HOLD", "BUY"][action],
                        "confidence": float(confidence),
                        "probabilities": probabilities.tolist(),
                        "adaptation_used": False
                    }
            except Exception as exc:
                self.logger.error(f"Base model prediction failed for {pair}: {exc}")
                return {"pair": pair, "regime": regime, "prediction_failed": True, "error": str(exc)}
        
        try:
            adapted_model, adapt_info = self.adapted_models[model_key]
            
            with torch.no_grad():
                x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                predictions = adapted_model(x)
                probabilities = torch.softmax(predictions, dim=1).cpu().numpy()[0]
                
                action = np.argmax(probabilities)
                confidence = np.max(probabilities)
                
                # Combine adaptation confidence with prediction confidence
                combined_confidence = (confidence + adapt_info["confidence"]) / 2
                
                return {
                    "pair": pair,
                    "regime": regime,
                    "action": ["SELL", "HOLD", "BUY"][action],
                    "confidence": float(combined_confidence),
                    "probabilities": probabilities.tolist(),
                    "adaptation_used": True,
                    "adaptation_accuracy": adapt_info["final_accuracy"]
                }
                
        except Exception as exc:
            self.logger.error(f"Adapted model prediction failed for {pair}: {exc}")
            return {"pair": pair, "regime": regime, "prediction_failed": True, "error": str(exc)}
    
    def meta_train_on_tasks(self, num_tasks: Optional[int] = None) -> Dict[str, Any]:
        """Perform meta-training on available tasks."""
        
        task_count = min(num_tasks or self.config.max_tasks, len(self.market_tasks))
        
        if task_count == 0:
            return {"error": "No tasks available for meta-training"}
        
        # Sample tasks for meta-training
        task_names = list(self.market_tasks.keys())[-task_count:]
        task_batch = []
        
        for task_name in task_names:
            task = self.market_tasks[task_name]
            task_batch.append((
                task["support_data"], task["support_labels"],
                task["query_data"], task["query_labels"]
            ))
        
        try:
            # Perform meta-update
            meta_metrics = self.meta_learner.meta_update(task_batch)
            
            self.logger.info(
                f"Meta-train completed - "
                f"Loss: {meta_metrics['meta_loss']:.4f} - "
                f"Accuracy: {meta_metrics['meta_accuracy']:.2%}"
            )
            
            return {
                "meta_train_successful": True,
                "tasks_used": len(task_batch),
                **meta_metrics
            }
            
        except Exception as exc:
            self.logger.error(f"Meta-training failed: {exc}")
            return {"meta_train_successful": False, "error": str(exc)}
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get statistics about model adaptations."""
        
        if not self.adaptation_history:
            return {"total_adaptations": 0}
        
        # Group adaptations by pair
        pair_stats: Dict[str, Dict[str, Any]] = {}
        regime_stats: Dict[str, int] = {}
        
        for record in self.adaptation_history:
            pair = record["pair"]
            regime = record["regime"]
            
            if pair not in pair_stats:
                pair_stats[pair] = {
                    "adaptations": 0,
                    "avg_accuracy": 0,
                    "avg_confidence": 0
                }
            
            pair_stats[pair]["adaptations"] += 1
            pair_stats[pair]["avg_accuracy"] += record["adaptation_info"]["final_accuracy"]
            pair_stats[pair]["avg_confidence"] += record["adaptation_info"]["confidence"]
            
            regime_stats[regime] = regime_stats.get(regime, 0) + 1
        
        # Calculate averages
        for stats in pair_stats.values():
            stats["avg_accuracy"] /= stats["adaptations"]
            stats["avg_confidence"] /= stats["adaptations"]
        
        return {
            "total_adaptations": len(self.adaptation_history),
            "pairs_adapted": len(pair_stats),
            "regimes_trained": len(regime_stats),
            "pair_stats": pair_stats,
            "regime_stats": regime_stats,
            "recent_adaptations": [
                {
                    "pair": r["pair"],
                    "regime": r["regime"],
                    "accuracy": r["adaptation_info"]["final_accuracy"],
                    "timestamp": r["timestamp"].isoformat()
                }
                for r in self.adaptation_history[-5:]
            ]
        }
    
    def clear_adaptations(self, pair: Optional[str] = None) -> int:
        """Clear adapted models."""
        if pair:
            # Clear adaptations for specific pair
            keys_to_remove = [k for k in self.adapted_models.keys() if k.startswith(f"{pair}_")]
            count = len(keys_to_remove)
            for key in keys_to_remove:
                del self.adapted_models[key]
        else:
            # Clear all adaptations
            count = len(self.adapted_models)
            self.adapted_models.clear()
        
        self.logger.info(f"Cleared {count} model adaptations")
        return count
    
    def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """Clean up old tasks to prevent memory buildup."""
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        old_tasks = [
            name for name, task in self.market_tasks.items()
            if task["created_at"] < cutoff_time
        ]
        
        for task_name in old_tasks:
            del self.market_tasks[task_name]
        
        if old_tasks:
            self.logger.info(f"Cleaned up {len(old_tasks)} old tasks")
        
        return len(old_tasks)