"""RiverLagNet CrossFormer: temporal Transformer with directed lag GNN fusion."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from RiverLagNet.data.graph_builder import build_graph_variant, expand_directed_paths

from .decoder import MultiHorizonMultiTargetDecoder, UpstreamResidualDecoder
from .forecast_transport_fusion import TargetConditionedForecastTransport
from .graph_cross_attention import (
    EdgeLagHorizonSparseAttention,
    TransformerGraphCrossFusion,
)
from .history_propagation import DirectedLaggedHistoryPropagation
from .input_encoder import InputMaskEncoder
from .temporal_transformer import NodeTemporalTransformer


class RiverGraphCrossFormer(nn.Module):
    """Fuse temporal Transformer queries with directed edge-lag GNN tokens.

    The model has two coupled innovations:

    1. Multi-scale Ancestor-Path Lag Attention jointly selects directed paths
       and causally observable travel lags for each forecast lead.
    2. Dual-space Transformer-GNN Fusion combines hidden routing-head tokens
       and target-conditioned upstream forecast transport in zero-started
       residual corrections.
    """

    def __init__(
        self,
        value_dim: int,
        static_dim: int,
        time_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        output_window: int = 30,
        target_dim: int = 3,
        max_lag: int = 30,
        graph_variant: str = "directed",
        graph_seed: int = 42,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        graph_heads: int = 4,
        history_steps: int = 8,
        max_path_hops: int = 8,
        dropout: float = 0.1,
        lag_prior_scale_days: float = 2.0,
        **_: object,
    ) -> None:
        super().__init__()
        if graph_variant not in {"directed", "undirected", "shuffled", "no_graph"}:
            raise ValueError("unsupported graph_variant")
        if max_lag < output_window:
            raise ValueError("max_lag must cover every direct forecast horizon")
        self.graph_variant = graph_variant
        self.graph_seed = graph_seed
        self.output_window = output_window
        self.max_path_hops = max_path_hops
        self.input_encoder = InputMaskEncoder(
            value_dim, static_dim, time_dim, hidden_dim
        )
        self.history_diffusion = DirectedLaggedHistoryPropagation(
            hidden_dim,
            edge_dim,
            max_lag,
            steps=history_steps,
            dropout=dropout,
        )
        self.temporal_transformer = NodeTemporalTransformer(
            hidden_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            dropout=dropout,
        )
        self.local_decoder = MultiHorizonMultiTargetDecoder(
            hidden_dim, output_window, target_dim
        )
        self.graph_attention = EdgeLagHorizonSparseAttention(
            hidden_dim,
            edge_dim,
            num_heads=graph_heads,
            max_lag=max_lag,
            max_path_hops=max_path_hops,
            prior_scale_days=lag_prior_scale_days,
        )
        self.cross_fusion = TransformerGraphCrossFusion(hidden_dim, graph_heads)
        self.forecast_transport = TargetConditionedForecastTransport(
            hidden_dim, target_dim, graph_heads
        )
        self.upstream_decoder = UpstreamResidualDecoder(hidden_dim, target_dim)
        with torch.no_grad():
            for head in self.upstream_decoder.heads:
                head.weight.zero_()
        self.attention_weights: Tensor | None = None
        self.fusion_weights: Tensor | None = None
        self.history_routing: Tensor | None = None
        self._upstream_residual_only = False
        self._expanded_graph_cache: tuple[Tensor, Tensor, Tensor] | None = None

    def configure_upstream_residual_training(self, gate_bias: float = -2.0) -> None:
        """Freeze the local Transformer and train only graph innovations."""
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for module in (
            self.history_diffusion,
            self.graph_attention,
            self.cross_fusion,
            self.forecast_transport,
            self.upstream_decoder,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        with torch.no_grad():
            self.cross_fusion.gate.weight.zero_()
            self.cross_fusion.gate.bias.fill_(gate_bias)
            for head in self.upstream_decoder.heads:
                head.weight.zero_()
        self._upstream_residual_only = True

    def train(self, mode: bool = True) -> RiverGraphCrossFormer:
        """Keep the frozen local Transformer deterministic in residual training."""
        super().train(mode)
        if mode and self._upstream_residual_only:
            for module in (
                self.input_encoder,
                self.temporal_transformer,
                self.local_decoder,
            ):
                module.eval()
        return self

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
        """Return ``y_hat [B,T_out,N,3]`` without flattening public axes."""
        encoded = self.input_encoder(x, x_mask, x_quality, static, time_features)
        if self.graph_variant == "no_graph":
            graph_encoded = encoded
            variant_edges = None
            variant_attr = None
            self.history_routing = None
        else:
            variant_edges, variant_attr = build_graph_variant(
                edge_index, edge_attr, self.graph_variant, self.graph_seed
            )
            graph_encoded, self.history_routing = self.history_diffusion(
                encoded, variant_edges, variant_attr
            )
        history_states, local_state = self.temporal_transformer(graph_encoded)
        local_context = self.local_decoder.contextualize(local_state)
        local_prediction = self.local_decoder.decode_context(local_context)
        if self.graph_variant == "no_graph":
            self.attention_weights = None
            self.fusion_weights = None
            return local_prediction

        assert variant_edges is not None and variant_attr is not None
        attention_edges, attention_attr, path_hops = self._expanded_graph(
            variant_edges, variant_attr
        )
        graph_heads, self.attention_weights = self.graph_attention(
            history_states,
            local_context,
            local_context,
            attention_edges,
            attention_attr,
            path_hops,
        )
        graph_state, self.fusion_weights = self.cross_fusion(
            local_context, graph_heads
        )
        target_correction = self.forecast_transport(
            x[..., : local_prediction.shape[-1]],
            x_mask[..., : local_prediction.shape[-1]],
            local_prediction,
            local_context,
            graph_state,
            attention_edges,
            self.attention_weights,
        )
        return (
            local_prediction
            + self.upstream_decoder(graph_state)
            + target_correction
        )

    def _expanded_graph(
        self, edge_index: Tensor, edge_attr: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return cached one-to-K-hop directed path candidates."""
        if self._expanded_graph_cache is None:
            self._expanded_graph_cache = expand_directed_paths(
                edge_index,
                edge_attr,
                max_hops=self.max_path_hops,
            )
        cached_index, cached_attr, cached_hops = self._expanded_graph_cache
        if cached_index.device != edge_index.device:
            cached_index = cached_index.to(edge_index.device)
            cached_attr = cached_attr.to(edge_attr.device)
            cached_hops = cached_hops.to(edge_index.device)
            self._expanded_graph_cache = (cached_index, cached_attr, cached_hops)
        return cached_index, cached_attr, cached_hops
