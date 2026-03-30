"""
Training Engine Module for Auto-Improver.
Trains ML models and optimizes strategy parameters.
"""

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from reco_trading.auto_improver.data_collector import DataSet, MarketDataPoint

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Result of a training run."""
    model_name: str
    metrics: dict[str, float]
    parameters: dict[str, Any]
    training_time: float
    dataset_name: str
    status: str
    error_message: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SignalLabeler:
    """Labels market data with trading signals."""

    @staticmethod
    def label_signals(data_points: list[MarketDataPoint], lookahead: int = 5) -> tuple[list, list]:
        """Label data points with buy/sell/hold signals."""
        closes = np.array([p.close for p in data_points])
        
        X = []
        y = []
        
        for i in range(lookahead, len(closes)):
            features = [
                closes[i] - closes[i-1],
                closes[i] - closes[i-5] if i >= 5 else 0,
                np.std(closes[max(0, i-10):i]) if i >= 10 else 0,
                (closes[i] - np.mean(closes[max(0, i-20):i])) / np.std(closes[max(0, i-20):i]) if i >= 20 else 0,
            ]
            
            future_return = (closes[i + lookahead] - closes[i]) / closes[i] if i + lookahead < len(closes) else 0
            
            if future_return > 0.02:
                label = 1
            elif future_return < -0.02:
                label = -1
            else:
                label = 0
            
            X.append(features)
            y.append(label)
        
        return X, y


class TrainingEngine:
    """Engine for training ML models and optimizing strategies."""

    def __init__(self, models_dir: Path | None = None):
        self.models_dir = models_dir or Path("./user_data/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self._trained_models: dict[str, TrainingResult] = {}

    async def train_signal_predictor(
        self,
        dataset: DataSet,
        model_name: str = "signal_predictor",
    ) -> TrainingResult:
        """Train a signal prediction model."""
        logger.info(f"Training signal predictor on dataset {dataset.name}")
        
        start_time = datetime.now(timezone.utc)
        
        try:
            all_data = dataset.data_points
            symbol_data: dict[str, list[MarketDataPoint]] = {}
            
            for point in all_data:
                if point.symbol not in symbol_data:
                    symbol_data[point.symbol] = []
                symbol_data[point.symbol].append(point)
            
            all_X = []
            all_y = []
            
            for symbol, points in symbol_data.items():
                points_sorted = sorted(points, key=lambda p: p.timestamp)
                X, y = SignalLabeler.label_signals(points_sorted)
                all_X.extend(X)
                all_y.extend(y)
            
            if len(all_X) < 100:
                return TrainingResult(
                    model_name=model_name,
                    metrics={},
                    parameters={},
                    training_time=0,
                    dataset_name=dataset.name,
                    status="failed",
                    error_message="Insufficient data for training",
                )
            
            X_train = np.array(all_X)
            y_train = np.array(all_y)
            
            from sklearn.model_selection import train_test_split
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            model = self._create_model()
            model.fit(X_tr, y_tr)
            
            train_acc = model.score(X_tr, y_tr)
            val_acc = model.score(X_val, y_val)
            
            model_path = self.models_dir / f"{model_name}.json"
            self._save_model(model, model_path)
            
            training_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            result = TrainingResult(
                model_name=model_name,
                metrics={
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "samples": len(all_X),
                },
                parameters={"lookahead": 5},
                training_time=training_time,
                dataset_name=dataset.name,
                status="success",
            )
            
            self._trained_models[model_name] = result
            
            logger.info(f"Training complete: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")
            return result
            
        except ImportError:
            return TrainingResult(
                model_name=model_name,
                metrics={},
                parameters={},
                training_time=0,
                dataset_name=dataset.name,
                status="skipped",
                error_message="sklearn not available - using mock training",
            )
        except Exception as e:
            logger.exception(f"Training failed: {e}")
            return TrainingResult(
                model_name=model_name,
                metrics={},
                parameters={},
                training_time=0,
                dataset_name=dataset.name,
                status="failed",
                error_message=str(e),
            )

    def _create_model(self):
        """Create ML model."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        except ImportError:
            return None

    def _save_model(self, model: Any, path: Path) -> None:
        """Save model metadata."""
        metadata = {
            "path": str(path),
            "type": type(model).__name__ if model else "mock",
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    async def optimize_strategy_parameters(
        self,
        base_params: dict[str, Any],
        dataset: DataSet,
        objective: str = "sharpe_ratio",
        iterations: int = 50,
    ) -> dict[str, Any]:
        """Optimize strategy parameters using random search."""
        logger.info(f"Optimizing strategy parameters ({iterations} iterations)")
        
        best_params = base_params.copy()
        best_score = float("-inf")
        
        param_ranges = {
            "rsi_period": (7, 21),
            "rsi_overbought": (65, 85),
            "rsi_oversold": (15, 35),
            "ma_period": (10, 50),
            "stop_loss": (0.01, 0.1),
            "take_profit": (0.02, 0.15),
        }
        
        for i in range(iterations):
            params = base_params.copy()
            
            for param, (min_val, max_val) in param_ranges.items():
                if isinstance(min_val, int):
                    params[param] = random.randint(min_val, max_val)
                else:
                    params[param] = random.uniform(min_val, max_val)
            
            score = await self._evaluate_params(params, dataset)
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
                logger.info(f"New best: {objective}={score:.3f}")
        
        logger.info(f"Optimization complete: best {objective}={best_score:.3f}")
        return best_params

    async def _evaluate_params(self, params: dict[str, Any], dataset: DataSet) -> float:
        """Evaluate parameters on dataset (mock implementation)."""
        return random.uniform(-1, 3)

    def get_trained_models(self) -> list[TrainingResult]:
        """Get list of trained models."""
        return list(self._trained_models.values())

    async def retrain_model(
        self,
        model_name: str,
        new_dataset: DataSet,
    ) -> TrainingResult:
        """Retrain an existing model with new data."""
        return await self.train_signal_predictor(new_dataset, model_name)
