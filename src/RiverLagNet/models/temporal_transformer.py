"""Node-wise temporal Transformer for multivariate observation histories."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class NodeTemporalTransformer(nn.Module):
    """Encode each node's complete past window with shared self-attention.

    Attention is bidirectional only inside the already observed input window;
    no forecast target or timestamp after the forecast origin is supplied.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_multiplier: int = 3,
        dropout: float = 0.1,
        max_history: int = 512,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if num_layers <= 0 or feedforward_multiplier <= 0 or max_history <= 0:
            raise ValueError("layer, feedforward, and history sizes must be positive")
        self.hidden_dim = hidden_dim
        self.max_history = max_history
        self.register_buffer(
            "position_encoding",
            _sinusoidal_encoding(max_history, hidden_dim),
            persistent=False,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * feedforward_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
            enable_nested_tensor=False,
        )

    def forward(self, encoded: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``h_seq [B,T,N,D]`` and ``h_local [B,N,D]``."""
        if encoded.ndim != 4 or encoded.shape[-1] != self.hidden_dim:
            raise ValueError("encoded must have shape [B,T,N,hidden_dim]")
        batch, history, nodes, features = encoded.shape
        if history > self.max_history:
            raise ValueError("history exceeds configured positional encoding length")
        sequences = encoded.permute(0, 2, 1, 3).reshape(
            batch * nodes, history, features
        )
        positions = self.position_encoding[:history].to(
            device=encoded.device, dtype=encoded.dtype
        )
        output = self.encoder(sequences + positions[None])
        h_seq = output.reshape(batch, nodes, history, features).permute(0, 2, 1, 3)
        return h_seq, h_seq[:, -1]


def _sinusoidal_encoding(length: int, hidden_dim: int) -> Tensor:
    position = torch.arange(length, dtype=torch.float32)[:, None]
    scale = torch.exp(
        torch.arange(0, hidden_dim, 2, dtype=torch.float32)
        * (-math.log(10_000.0) / hidden_dim)
    )
    encoding = torch.zeros(length, hidden_dim, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * scale)
    encoding[:, 1::2] = torch.cos(position * scale[: encoding[:, 1::2].shape[1]])
    return encoding
