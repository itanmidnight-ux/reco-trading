from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(slots=True)
class OrderBookSnapshot:
    """Contrato de entrada por timestamp para el transformer.

    Campos esperados por snapshot:
    - `book_features`: vector con features normalizadas del order book (niveles bid/ask).
    - `obi`: Order Book Imbalance en [-1, 1].
    - `cvd`: Cumulative Volume Delta normalizado.
    - `spread`: spread relativo o en ticks normalizado.
    - `trade_print_features`: agregados de trade prints (p. ej. buy ratio, volumen, intensidad).
    """

    book_features: list[float]
    obi: float
    cvd: float
    spread: float
    trade_print_features: list[float]

    def to_feature_vector(self) -> list[float]:
        return [*self.book_features, self.obi, self.cvd, self.spread, *self.trade_print_features]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class OrderFlowTransformer(nn.Module):
    """Transformer encoder para dirección short-term (up/down)."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim debe ser divisible por num_heads")

        self.input_projection = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, input_dim] -> prob_up: [batch]."""
        if x.ndim != 3:
            raise ValueError("Input esperado con forma [batch, seq_len, input_dim]")

        z = self.input_projection(x)
        z = self.positional_encoding(z)
        z = self.encoder(z)
        z = self.norm(z)
        pooled = z[:, -1, :]
        logits = self.classifier(pooled).squeeze(-1)
        return torch.sigmoid(logits)


@dataclass(slots=True)
class InferenceResult:
    transformer_prob_up: float
    latency_ms: float
    timed_out: bool


class OrderFlowInferenceAdapter:
    """Adaptador async para inferencia con límite de latencia."""

    def __init__(
        self,
        model: OrderFlowTransformer,
        latency_budget_ms: float = 20.0,
        fallback_probability: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.latency_budget_ms = float(max(latency_budget_ms, 1.0))
        self.fallback_probability = float(np.clip(fallback_probability, 0.0, 1.0))

    def _predict_sync(self, sequence_features: np.ndarray | torch.Tensor) -> float:
        if isinstance(sequence_features, np.ndarray):
            x = torch.from_numpy(sequence_features).float()
        else:
            x = sequence_features.float()

        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise ValueError("sequence_features debe ser [seq_len, feat_dim] o [batch, seq_len, feat_dim]")

        x = x.to(self.device)
        with torch.no_grad():
            probs = self.model(x)
        return float(torch.clamp(probs[0], 0.0, 1.0).item())

    async def infer_transformer_prob_up(self, sequence_features: np.ndarray | torch.Tensor) -> InferenceResult:
        start = time.perf_counter()
        try:
            prob = await asyncio.wait_for(
                asyncio.to_thread(self._predict_sync, sequence_features),
                timeout=self.latency_budget_ms / 1000.0,
            )
            latency_ms = (time.perf_counter() - start) * 1000.0
            return InferenceResult(transformer_prob_up=prob, latency_ms=latency_ms, timed_out=False)
        except TimeoutError:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return InferenceResult(
                transformer_prob_up=self.fallback_probability,
                latency_ms=latency_ms,
                timed_out=True,
            )
