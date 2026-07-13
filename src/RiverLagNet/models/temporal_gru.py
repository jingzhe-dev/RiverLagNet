"""Node-independent temporal GRU encoder."""

from __future__ import annotations

from torch import Tensor, nn


class NodeTemporalGRU(nn.Module):
    """Encode each station history independently while retaining every hidden state."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, encoded: Tensor) -> tuple[Tensor, Tensor]:
        """Return `h_seq [B,T,N,D]` and `h_local [B,N,D]`."""
        if encoded.ndim != 4:
            raise ValueError("encoded must have shape [B,T,N,D]")
        batch, history, nodes, features = encoded.shape
        node_sequences = encoded.permute(0, 2, 1, 3).reshape(batch * nodes, history, features)
        output, _ = self.gru(node_sequences)
        h_seq = output.reshape(batch, nodes, history, self.hidden_dim).permute(0, 2, 1, 3)
        return h_seq, h_seq[:, -1]
