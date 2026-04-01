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
class TFTConfig:
    """Configuration for Temporal Fusion Transformer."""
    hidden_size: int = 64
    dropout: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    sequence_length: int = 60
    num_heads: int = 4
    num_layers: int = 2
    prediction_horizon: int = 5
    early_stopping_patience: int = 10


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network for TFT."""
    
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.linear = nn.Linear(input_size, hidden_size)
        self.gate = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_out = self.linear(x)
        gate_out = self.gate(x)
        
        gated = linear_out * gate_out
        gated = self.dropout(gated)
        
        # Add skip connection if dimensions match
        if x.shape[-1] == self.hidden_size:
            return self.layer_norm(gated + x)
        return self.layer_norm(gated)


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for TFT."""
    
    def __init__(self, input_size: int, hidden_size: int, num_inputs: int, dropout: float = 0.1):
        super().__init__()
        self.num_inputs = num_inputs
        
        # Flattened input size
        self.flattened_size = input_size * num_inputs
        
        self.sparse_weights = nn.Sequential(
            nn.Linear(self.flattened_size, num_inputs),
            nn.Softmax(dim=-1)
        )
        
        self.grns = nn.ModuleList([
            GatedResidualNetwork(input_size, hidden_size, dropout)
            for _ in range(num_inputs)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, num_inputs, input_size)
        batch_size, seq_len, num_inputs, input_size = x.shape
        
        # Flatten for sparse weights computation
        x_flat = x.reshape(batch_size, seq_len, -1)
        
        # Compute sparse weights
        sparse_weights = self.sparse_weights(x_flat)
        sparse_weights = sparse_weights.unsqueeze(-1)
        
        # Apply GRNs to each input
        transformed_inputs = []
        for i in range(num_inputs):
            transformed = self.grns[i](x[:, :, i, :])
            transformed_inputs.append(transformed.unsqueeze(2))
        
        transformed_inputs = torch.cat(transformed_inputs, dim=2)
        
        # Weighted combination
        weighted_inputs = transformed_inputs * sparse_weights
        output = torch.sum(weighted_inputs, dim=2)
        
        return output


class InterpretableMultiHeadAttention(nn.Module):
    """Interpretable Multi-Head Attention for TFT."""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Attention output
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Add & norm
        output = self.layer_norm(context + x)
        
        return output, attention_weights


class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer for time series forecasting."""
    
    def __init__(self, config: TFTConfig, input_size: int, output_size: int = 1):
        super().__init__()
        self.config = config
        
        # Input embedding
        self.input_embedding = nn.Linear(input_size, config.hidden_size)
        
        # Variable selection
        self.static_vsn = VariableSelectionNetwork(
            input_size=input_size, 
            hidden_size=config.hidden_size, 
            num_inputs=4,  # Assume 4 static variables (open, high, low, volume)
            dropout=config.dropout
        )
        
        self.future_vsn = VariableSelectionNetwork(
            input_size=input_size, 
            hidden_size=config.hidden_size, 
            num_inputs=config.prediction_horizon,
            dropout=config.dropout
        )
        
        self.historical_vsn = VariableSelectionNetwork(
            input_size=input_size, 
            hidden_size=config.hidden_size, 
            num_inputs=config.sequence_length,
            dropout=config.dropout
        )
        
        # Encoder (self-attention for historical data)
        self.encoder_attention = nn.ModuleList([
            InterpretableMultiHeadAttention(config.hidden_size, config.num_heads, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        # Context gate for future selection
        self.context_gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Sigmoid()
        )
        
        # Decoder (self-attention for future data with context)
        self.decoder_attention = nn.ModuleList([
            InterpretableMultiHeadAttention(config.hidden_size, config.num_heads, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        # Position-wise feed-forward
        self.position_wise = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size * 4),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size * 4, config.hidden_size),
                nn.Dropout(config.dropout)
            )
            for _ in range(config.num_layers * 2)  # Encoder + decoder layers
        ])
        
        # Final gating and output
        self.gate_add_norm = nn.LayerNorm(config.hidden_size)
        self.output_layer = nn.Linear(config.hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, num_features)
            
        Returns:
            predictions: Predicted values of shape (batch_size, prediction_horizon)
            attention_weights: Attention weights for interpretability
        """
        batch_size, seq_len, num_features = x.shape
        
        # Input embedding
        embedded = self.input_embedding(x)
        
        # Split into historical and future
        historical = embedded[:, :-self.config.prediction_horizon, :]
        future = embedded[:, -self.config.prediction_horizon:, :]
        
        # Variable selection networks
        static_context = self.static_vsn(embedded)
        historical_encoded = self.historical_vsn(historical)
        future_encoded = self.future_vsn(future)
        
        # Encoder
        encoder_out = historical_encoded
        attention_weights = None
        
        for i, attention_layer in enumerate(self.encoder_attention):
            attn_out, attn_weights = attention_layer(encoder_out)
            encoder_out = self.position_wise[i](attn_out)
            attention_weights = attn_weights
        
        # Context for future selection
        encoder_context = encoder_out[:, -1, :]  # Last time step
        
        # Gate for future selection
        gate = self.context_gate(torch.cat([static_context[:, -1, :], encoder_context], dim=-1))
        gated_context = gate * encoder_context
        
        # Prepare future input with context
        future_input = torch.cat([
            future_encoded,
            gated_context.unsqueeze(1).expand(-1, self.config.prediction_horizon, -1)
        ], dim=-1)
        
        # Decoder
        decoder_out = future_input
        
        for i, attention_layer in enumerate(self.decoder_attention):
            attn_out, _ = attention_layer(decoder_out)
            decoder_out = self.position_wise[self.config.num_layers + i](attn_out)
        
        # Gate addition & norm
        gated_add = decoder_out * gate.unsqueeze(1)
        output = self.gate_add_norm(gated_add + future_encoded)
        
        # Final output
        predictions = self.output_layer(output)
        
        return predictions, attention_weights


class TFTManager:
    """Manager for Temporal Fusion Transformer models."""
    
    def __init__(self, config: Optional[TFTConfig] = None):
        self.config = config or TFTConfig()
        self.logger = logging.getLogger(__name__)
        self.models: Dict[str, Tuple[TemporalFusionTransformer, TFTConfig]] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        target_col: str = "close",
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Prepare data for TFT training."""
        
        if df.empty:
            raise ValueError("Empty dataframe provided")
        
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col in ["open", "high", "low", "close", "volume"]]
        
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        if not feature_cols:
            raise ValueError("No valid feature columns found")
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(len(df) - self.config.sequence_length - self.config.prediction_horizon + 1):
            # Input sequence
            seq = df[feature_cols].iloc[i:i + self.config.sequence_length].values
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
        
        return X, y, feature_cols
    
    def train(
        self, 
        df: pd.DataFrame, 
        pair: str = "BTC/USDT",
        target_col: str = "close"
    ) -> Dict[str, Any]:
        """Train TFT model for a specific pair."""
        
        try:
            X, y, feature_cols = self.prepare_data(df, target_col)
            
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
            model = TemporalFusionTransformer(
                config=self.config,
                input_size=len(feature_cols),
                output_size=self.config.prediction_horizon
            ).to(self.device)
            
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
                    predictions, _ = model(batch_X)
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
                        predictions, _ = model(batch_X)
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
                final_predictions, attention_weights = model(X_test.to(self.device))
                final_loss = criterion(final_predictions, y_test.to(self.device))
            
            self.logger.info(
                f"TFT model trained for {pair} - "
                f"Final Loss: {final_loss.item():.6f} - "
                f"Best Loss: {best_loss:.6f}"
            )
            
            return {
                "pair": pair,
                "final_loss": final_loss.item(),
                "best_loss": best_loss,
                "feature_cols": feature_cols,
                "sequence_length": self.config.sequence_length,
                "prediction_horizon": self.config.prediction_horizon,
                "samples": len(X_train)
            }
            
        except Exception as exc:
            self.logger.error(f"TFT training failed for {pair}: {exc}")
            raise
    
    def predict(
        self, 
        df: pd.DataFrame, 
        pair: str = "BTC/USDT"
    ) -> Optional[Dict[str, Any]]:
        """Make predictions using trained TFT model."""
        
        if pair not in self.models:
            self.logger.warning(f"No trained model for {pair}")
            return None
        
        try:
            model, config = self.models[pair]
            model.eval()
            
            X, _, feature_cols = self.prepare_data(df)
            
            if len(X) < 1:
                return None
            
            with torch.no_grad():
                X = X.to(self.device)
                predictions, attention_weights = model(X[-1:])  # Last sequence
                
                predictions = predictions.cpu().numpy().flatten()
                
                # Calculate confidence based on attention weights
                if attention_weights is not None:
                    attention_confidence = torch.mean(attention_weights).item()
                else:
                    attention_confidence = 0.5
            
            return {
                "predictions": predictions.tolist(),
                "confidence": float(attention_confidence),
                "feature_cols": feature_cols,
                "prediction_horizon": config.prediction_horizon,
                "direction": "BUY" if predictions[0] > predictions[1] else "SELL",
                "predicted_change_pct": float((predictions[0] - predictions[1]) / predictions[1] * 100)
            }
            
        except Exception as exc:
            self.logger.error(f"TFT prediction failed for {pair}: {exc}")
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
                "num_heads": config.num_heads,
                "num_layers": config.num_layers
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
        self.logger.info("All TFT models cleared")