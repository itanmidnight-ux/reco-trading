import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


@dataclass
class NBEATSConfig:
    """Configuration for N-BEATS model."""
    hidden_size: int = 512
    num_stacks: int = 30
    num_blocks: int = 3
    num_layers: int = 4
    theta_dim: List[int] = field(default_factory=lambda: [8, 8])
    dropout: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    sequence_length: int = 60
    prediction_horizon: int = 5
    early_stopping_patience: int = 10
    share_weights: bool = False


class SeasonalityBlock(nn.Module):
    """Seasonality block for N-BEATS."""
    
    def __init__(self, input_size: int, theta_dim: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        
        layers = []
        in_size = input_size
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_size, input_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_size = input_size
        
        layers.append(nn.Linear(in_size, theta_dim * 2))
        
        self.stack = nn.Sequential(*layers)
        self.theta_dim = theta_dim
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input shape: (batch_size, input_size)
        theta = self.stack(x)
        theta_f, theta_b = theta[:, :self.theta_dim], theta[:, self.theta_dim:]
        
        # Seasonality basis (sine/cosine waves)
        backcast = self._seasonality_backcast(x, theta_b)
        forecast = self._seasonality_forecast(theta_f)
        
        return backcast, forecast
    
    def _seasonality_backcast(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Seasonality backcast."""
        # Simple linear combination with basis functions
        return torch.matmul(x, theta.t())
    
    def _seasonality_forecast(self, theta: torch.Tensor) -> torch.Tensor:
        """Seasonality forecast."""
        # Linear projection to forecast space
        return theta.unsqueeze(1)


class TrendBlock(nn.Module):
    """Trend block for N-BEATS."""
    
    def __init__(self, input_size: int, theta_dim: int, num_layers: int, forecast_horizon: int = 5, dropout: float = 0.1):
        super().__init__()
        
        layers = []
        in_size = input_size
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_size, input_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_size = input_size
        
        layers.append(nn.Linear(in_size, theta_dim * 2))
        
        self.stack = nn.Sequential(*layers)
        self.theta_dim = theta_dim
        self.forecast_horizon = forecast_horizon
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input shape: (batch_size, input_size)
        theta = self.stack(x)
        theta_f, theta_b = theta[:, :self.theta_dim], theta[:, self.theta_dim:]
        
        # Trend basis (polynomial)
        backcast = self._trend_backcast(x, theta_b)
        forecast = self._trend_forecast(theta_f)
        
        return backcast, forecast
    
    def _trend_backcast(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Trend backcast using polynomial basis."""
        batch_size = x.size(0)
        time_steps = x.size(1)
        
        # Create polynomial basis
        t = torch.linspace(0, 1, time_steps, device=x.device)
        basis = torch.stack([t**i for i in range(self.theta_dim)])
        
        # Compute backcast
        backcast = torch.zeros(batch_size, time_steps, device=x.device)
        for i in range(self.theta_dim):
            backcast += theta[:, i:i+1] * basis[i]
        
        return backcast
    
    def _trend_forecast(self, theta: torch.Tensor) -> torch.Tensor:
        """Trend forecast using polynomial basis."""
        batch_size = theta.size(0)
        
        # Dynamic forecast based on configured horizon
        forecast = torch.zeros(batch_size, self.forecast_horizon, device=theta.device)
        for i in range(self.forecast_horizon):
            time_point = (i + 1) / self.forecast_horizon
            for j in range(self.theta_dim):
                forecast[:, i] += theta[:, j] * (time_point ** j)
        
        return forecast


class GenericBlock(nn.Module):
    """Generic block for N-BEATS."""
    
    def __init__(self, input_size: int, theta_dim: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        
        layers = []
        in_size = input_size
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_size, input_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_size = input_size
        
        layers.append(nn.Linear(in_size, theta_dim * 2))
        
        self.stack = nn.Sequential(*layers)
        self.theta_dim = theta_dim
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input shape: (batch_size, input_size)
        theta = self.stack(x)
        theta_f, theta_b = theta[:, :self.theta_dim], theta[:, self.theta_dim:]
        
        # Generic basis function
        backcast = self._generic_backcast(x, theta_b)
        forecast = self._generic_forecast(theta_f)
        
        return backcast, forecast
    
    def _generic_backcast(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Generic backcast."""
        # Simple linear transformation
        return torch.matmul(x, theta.t())
    
    def _generic_forecast(self, theta: torch.Tensor) -> torch.Tensor:
        """Generic forecast."""
        return theta.unsqueeze(1)


class Stack(nn.Module):
    """Stack of blocks in N-BEATS."""
    
    def __init__(
        self, 
        input_size: int, 
        block_type: str,
        num_blocks: int,
        theta_dim: int,
        num_layers: int,
        forecast_horizon: int = 5,
        dropout: float = 0.1,
        share_weights: bool = False
    ):
        super().__init__()
        
        self.blocks = nn.ParameterList()
        self.forecast_horizon = forecast_horizon
        
        if block_type == "trend":
            if share_weights:
                block = TrendBlock(input_size, theta_dim, num_layers, forecast_horizon, dropout)
                self.blocks.extend([block] * num_blocks)
            else:
                for _ in range(num_blocks):
                    self.blocks.append(TrendBlock(input_size, theta_dim, num_layers, forecast_horizon, dropout))
        elif block_type == "seasonality":
            if share_weights:
                block = SeasonalityBlock(input_size, theta_dim, num_layers, dropout)
                self.blocks.extend([block] * num_blocks)
            else:
                for _ in range(num_blocks):
                    self.blocks.append(SeasonalityBlock(input_size, theta_dim, num_layers, dropout))
        else:
            if share_weights:
                block = GenericBlock(input_size, theta_dim, num_layers, dropout)
                self.blocks.extend([block] * num_blocks)
            else:
                for _ in range(num_blocks):
                    self.blocks.append(GenericBlock(input_size, theta_dim, num_layers, dropout))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input shape: (batch_size, input_size)
        
        stack_forecast = 0
        backcast = x
        
        for block in self.blocks:
            block_backcast, block_forecast = block(backcast)
            backcast = backcast - block_backcast
            stack_forecast = stack_forecast + block_forecast
        
        return backcast, stack_forecast


class NBEATS(nn.Module):
    """N-BEATS: Neural Basis Expansion Analysis for Time Series Forecasting."""
    
    def __init__(self, config: NBEATSConfig):
        super().__init__()
        self.config = config
        
        stacks = []
        input_size = config.sequence_length
        
        # Create stacks with alternating block types
        for i in range(config.num_stacks):
            block_type = "generic"
            theta_dim = config.theta_dim[0] if i == 0 else config.theta_dim[1] if i == 1 else config.theta_dim[0]
            
            stack = Stack(
                input_size=input_size,
                block_type=block_type,
                num_blocks=config.num_blocks,
                theta_dim=theta_dim,
                num_layers=config.num_layers,
                forecast_horizon=config.prediction_horizon,
                dropout=config.dropout,
                share_weights=config.share_weights
            )
            
            stacks.append(stack)
        
        self.stacks = nn.ModuleList(stacks)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, sequence_length)
        
        total_forecast = 0
        backcast = x
        
        for stack in self.stacks:
            backcast, stack_forecast = stack(backcast)
            total_forecast += stack_forecast
        
        return total_forecast.squeeze()


class NBEATSManager:
    """Manager for N-BEATS models."""
    
    def __init__(self, config: Optional[NBEATSConfig] = None):
        self.config = config or NBEATSConfig()
        self.logger = logging.getLogger(__name__)
        self.models: Dict[str, Tuple[NBEATS, NBEATSConfig]] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        target_col: str = "close"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for N-BEATS training."""
        
        if df.empty:
            raise ValueError("Empty dataframe provided")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(len(df) - self.config.sequence_length - self.config.prediction_horizon + 1):
            # Input sequence
            seq = df[target_col].iloc[i:i + self.config.sequence_length].values
            sequences.append(seq)
            
            # Target (next prediction_horizon values)
            target = df[target_col].iloc[
                i + self.config.sequence_length:
                i + self.config.sequence_length + self.config.prediction_horizon
            ].values
            targets.append(target)
        
        if not sequences:
            raise ValueError("Not enough data to create sequences")
        
        X = torch.FloatTensor(np.array(sequences))
        y = torch.FloatTensor(np.array(targets))
        
        return X, y
    
    def train(
        self, 
        df: pd.DataFrame, 
        pair: str = "BTC/USDT",
        target_col: str = "close"
    ) -> Dict[str, Any]:
        """Train N-BEATS model for a specific pair."""
        
        try:
            X, y = self.prepare_data(df, target_col)
            
            # Split data
            train_size = int(0.8 * len(X))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Create datasets
            train_dataset = TensorDataset(X_train, y_train)
            test_dataset = TensorDataset(X_test, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)
            
            # Initialize model
            model = NBEATS(self.config).to(self.device)
            
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
            
            # Training loop
            best_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.config.epochs):
                model.train()
                train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    predictions = model(batch_X)
                    loss = criterion(predictions, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        predictions = model(batch_X)
                        loss = criterion(predictions, batch_y)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(test_loader)
                
                self.logger.info(
                    f"Epoch {epoch+1}/{self.config.epochs} - "
                    f"Train Loss: {train_loss/len(train_loader):.6f} - "
                    f"Val Loss: {avg_val_loss:.6f}"
                )
                
                # Early stopping
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    patience_counter = 0
                    
                    # Save best model
                    self.models[pair] = (model, self.config)
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Final evaluation
            model.eval()
            with torch.no_grad():
                final_predictions = model(X_test.to(self.device))
                final_loss = criterion(final_predictions, y_test.to(self.device))
            
            # Calculate directional accuracy
            last_values = X_test[:, -1].cpu().numpy()
            pred_directions = final_predictions[:, 0].cpu().numpy()
            actual_directions = y_test[:, 0].cpu().numpy()
            
            directional_accuracy = np.mean(
                (pred_directions > last_values) == (actual_directions > last_values)
            )
            
            self.logger.info(
                f"N-BEATS model trained for {pair} - "
                f"Final Loss: {final_loss.item():.6f} - "
                f"Best Loss: {best_loss:.6f} - "
                f"Directional Accuracy: {directional_accuracy:.2%}"
            )
            
            return {
                "pair": pair,
                "final_loss": final_loss.item(),
                "best_loss": best_loss,
                "directional_accuracy": directional_accuracy,
                "sequence_length": self.config.sequence_length,
                "prediction_horizon": self.config.prediction_horizon,
                "samples": len(X_train),
                "target_col": target_col
            }
            
        except Exception as exc:
            self.logger.error(f"N-BEATS training failed for {pair}: {exc}")
            raise
    
    def predict(
        self, 
        df: pd.DataFrame, 
        pair: str = "BTC/USDT",
        target_col: str = "close"
    ) -> Optional[Dict[str, Any]]:
        """Make predictions using trained N-BEATS model."""
        
        if pair not in self.models:
            self.logger.warning(f"No trained model for {pair}")
            return None
        
        try:
            model, config = self.models[pair]
            model.eval()
            
            X, _ = self.prepare_data(df, target_col)
            
            if len(X) < 1:
                return None
            
            with torch.no_grad():
                X = X.to(self.device)
                predictions = model(X[-1:])  # Last sequence
                
                predictions = predictions.cpu().numpy().flatten()
                
                # Get the last known value for comparison
                last_value = df[target_col].iloc[-1]
                
                # Calculate confidence based on prediction stability
                variance = np.var(predictions[:3]) if len(predictions) >= 3 else 0
                confidence = max(0.5, min(0.9, 1.0 - min(1.0, variance / (last_value * 0.01))))
            
            # Determine direction
            predicted_next = predictions[0]
            direction = "BUY" if predicted_next > last_value else "SELL"
            
            # Calculate percentage change
            if last_value > 0:
                predicted_change_pct = (predicted_next - last_value) / last_value * 100
            else:
                predicted_change_pct = 0
            
            return {
                "predictions": predictions.tolist(),
                "predicted_next": float(predicted_next),
                "last_value": float(last_value),
                "direction": direction,
                "predicted_change_pct": float(predicted_change_pct),
                "confidence": float(confidence),
                "sequence_length": config.sequence_length,
                "prediction_horizon": len(predictions)
            }
            
        except Exception as exc:
            self.logger.error(f"N-BEATS prediction failed for {pair}: {exc}")
            return None
    
    def get_model_info(self, pair: str) -> Optional[Dict[str, Any]]:
        """Get information about trained model."""
        if pair not in self.models:
            return None
        
        model, config = self.models[pair]
        
        return {
            "pair": pair,
            "config": {
                "hidden_size": config.hidden_size,
                "sequence_length": config.sequence_length,
                "prediction_horizon": config.prediction_horizon,
                "num_stacks": config.num_stacks,
                "num_blocks": config.num_blocks,
                "num_layers": config.num_layers,
                "theta_dim": config.theta_dim,
                "share_weights": config.share_weights
            },
            "device": str(self.device),
            "parameters": sum(p.numel() for p in model.parameters())
        }
    
    def clear_model(self, pair: str) -> bool:
        """Remove trained model for a pair."""
        if pair in self.models:
            del self.models[pair]
            return True
        return False
    
    def clear_all_models(self) -> None:
        """Remove all trained models."""
        self.models.clear()
        self.logger.info("All N-BEATS models cleared")