"""Complete RiverLagNet v0.1 forecasting architecture."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from RiverLagNet.data.graph_builder import build_graph_variant

from .decoder import MultiHorizonMultiTargetDecoder, UpstreamResidualDecoder
from .fusion import BoundedLinearHorizonGate, LocalUpstreamGatedFusion
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
        lag_prior_scale_days: float = 1.0,
        lag_prior_strength: float = 8.0,
        lag_residual_max_mix: float = 1.0,
        horizon_gate_mode: str = "none",
        **_: object,
    ) -> None:
        super().__init__()
        if graph_variant not in {"directed", "undirected", "shuffled", "no_graph"}:
            raise ValueError("unsupported graph_variant")
        if horizon_gate_mode not in {"none", "linear"}:
            raise ValueError("horizon_gate_mode must be none or linear")
        self.graph_variant = graph_variant
        self.graph_seed = graph_seed
        self.output_window = output_window
        self.horizon_gate_mode = horizon_gate_mode
        self._horizon_calibration_only = False
        self.input_encoder = InputMaskEncoder(value_dim, static_dim, time_dim, hidden_dim)
        self.temporal_encoder = NodeTemporalGRU(hidden_dim, hidden_dim)
        self.message_passing = DirectedLagAwareMessagePassing(
            hidden_dim,
            edge_dim,
            max_lag,
            lag_mode,
            dropout,
            lag_prior_scale_days,
            lag_prior_strength,
            lag_residual_max_mix,
        )
        self.fusion = LocalUpstreamGatedFusion(hidden_dim)
        self.decoder = MultiHorizonMultiTargetDecoder(hidden_dim, output_window, target_dim)
        self.upstream_decoder = UpstreamResidualDecoder(hidden_dim, target_dim)
        self.horizon_gate = (
            BoundedLinearHorizonGate(output_window)
            if horizon_gate_mode == "linear"
            else None
        )
        self.attention_weights: Tensor | None = None

    def configure_upstream_residual_training(self, gate_bias: float = -1.0) -> None:
        """Freeze the local forecaster and zero-start the upstream correction.

        The directed model initially reproduces its warm-started local
        prediction exactly. Only the message-passing, fusion, and upstream
        decoder parameters remain trainable, so any validation improvement is
        attributable to the upstream residual branch rather than a changed
        local backbone.
        """
        for module in (self.input_encoder, self.temporal_encoder, self.decoder):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        with torch.no_grad():
            self.fusion.gate.weight.zero_()
            self.fusion.gate.bias.fill_(gate_bias)
            for head in self.upstream_decoder.heads:
                head.weight.zero_()

    def configure_horizon_calibration_training(self) -> None:
        """Freeze the selected graph model and train only its two horizon scalars."""
        if self.horizon_gate is None:
            raise ValueError("horizon calibration requires horizon_gate_mode=linear")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.horizon_gate.parameters():
            parameter.requires_grad_(True)
        self._horizon_calibration_only = True

    def train(self, mode: bool = True) -> RiverLagNet:
        """Keep the frozen graph forecaster deterministic during calibration."""
        super().train(mode)
        if mode and self._horizon_calibration_only:
            for module in (
                self.input_encoder,
                self.temporal_encoder,
                self.message_passing,
                self.fusion,
                self.decoder,
                self.upstream_decoder,
            ):
                module.eval()
            assert self.horizon_gate is not None
            self.horizon_gate.train(True)
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
        """Return `y_hat [B,T_out,N,3]`."""
        encoded = self.input_encoder(x, x_mask, x_quality, static, time_features)
        h_seq, h_local = self.temporal_encoder(encoded)
        local_by_horizon = h_local[:, None].expand(-1, self.output_window, -1, -1)
        local_prediction = self.decoder(local_by_horizon)
        if self.graph_variant == "no_graph":
            self.attention_weights = None
            return local_prediction
        else:
            variant_edges, variant_attr = build_graph_variant(
                edge_index, edge_attr, self.graph_variant, self.graph_seed
            )
            h_upstream, attention = self.message_passing.forward_horizons(
                h_seq,
                h_local,
                variant_edges,
                variant_attr,
                output_window=self.output_window,
            )
            self.attention_weights = attention
            fused = self.fusion(local_by_horizon, h_upstream)
        upstream_state = fused - local_by_horizon
        upstream_correction = self.upstream_decoder(upstream_state)
        if self.horizon_gate is not None:
            upstream_correction = self.horizon_gate(upstream_correction)
        return local_prediction + upstream_correction
