"""Complete RiverLagNet v0.1 forecasting architecture."""

from __future__ import annotations

from torch import Tensor, nn

from RiverLagNet.data.graph_builder import build_graph_variant

from .decoder import MultiHorizonMultiTargetDecoder
from .fusion import LocalUpstreamGatedFusion
from .input_encoder import InputMaskEncoder
from .lag_message_passing import DirectedLagAwareMessagePassing
from .temporal_gru import NodeTemporalGRU


class RiverLagNet(nn.Module):
    """GRU forecaster centered on directed lag-aware upstream propagation."""

    def __init__(
        self,
        value_dim: int,
        static_dim: int,
        time_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        output_window: int = 30,
        target_dim: int = 3,
        max_lag: int = 14,
        graph_variant: str = "directed",
        lag_mode: str = "learned_lag",
        dropout: float = 0.0,
        graph_seed: int = 42,
        **_: object,
    ) -> None:
        super().__init__()
        if graph_variant not in {"directed", "undirected", "shuffled", "no_graph"}:
            raise ValueError("unsupported graph_variant")
        self.graph_variant = graph_variant
        self.graph_seed = graph_seed
        self.input_encoder = InputMaskEncoder(value_dim, static_dim, time_dim, hidden_dim)
        self.temporal_encoder = NodeTemporalGRU(hidden_dim, hidden_dim)
        self.message_passing = DirectedLagAwareMessagePassing(
            hidden_dim, edge_dim, max_lag, lag_mode, dropout
        )
        self.fusion = LocalUpstreamGatedFusion(hidden_dim)
        self.decoder = MultiHorizonMultiTargetDecoder(hidden_dim, output_window, target_dim)
        self.attention_weights: Tensor | None = None

    def forward(
        self,
        x: Tensor,
        x_mask: Tensor,
        x_quality: Tensor | None,
        static: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        time_features: Tensor,
    ) -> Tensor:
        """Return `y_hat [B,T_out,N,3]`."""
        encoded = self.input_encoder(x, x_mask, x_quality, static, time_features)
        h_seq, h_local = self.temporal_encoder(encoded)
        if self.graph_variant == "no_graph":
            self.attention_weights = None
            fused = h_local
        else:
            variant_edges, variant_attr = build_graph_variant(
                edge_index, edge_attr, self.graph_variant, self.graph_seed
            )
            h_upstream, attention = self.message_passing(
                h_seq, h_local, variant_edges, variant_attr
            )
            self.attention_weights = attention
            fused = self.fusion(h_local, h_upstream)
        return self.decoder(fused)
