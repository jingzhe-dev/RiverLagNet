"""Complete RiverLagNet v0.1 forecasting architecture."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from RiverLagNet.data.graph_builder import build_graph_variant

from .decoder import MultiHorizonMultiTargetDecoder, UpstreamResidualDecoder
from .fusion import BoundedLinearHorizonGate, LocalUpstreamGatedFusion
from .history_propagation import DirectedLaggedHistoryPropagation
from .input_encoder import InputMaskEncoder
from .lag_message_passing import DirectedLagAwareMessagePassing
from .output_transport import DirectedLaggedOutputTransport
from .temporal_gru import NodeTemporalGRU
from .topology_encoder import DirectedTopologyEncoder
from .trajectory_propagation import DirectedTrajectoryPropagation


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
        lag_bias_mode: str = "none",
        horizon_gate_mode: str = "none",
        propagation_mode: str = "legacy",
        trajectory_steps: int = 4,
        topology_mode: str = "none",
        **_: object,
    ) -> None:
        super().__init__()
        if graph_variant not in {"directed", "undirected", "shuffled", "no_graph"}:
            raise ValueError("unsupported graph_variant")
        if horizon_gate_mode not in {"none", "linear"}:
            raise ValueError("horizon_gate_mode must be none or linear")
        if propagation_mode not in {
            "legacy",
            "trajectory",
            "output_transport",
            "history",
        }:
            raise ValueError(
                "propagation_mode must be legacy, trajectory, output_transport, or history"
            )
        if topology_mode not in {"none", "structural"}:
            raise ValueError("topology_mode must be none or structural")
        self.graph_variant = graph_variant
        self.graph_seed = graph_seed
        self.output_window = output_window
        self.horizon_gate_mode = horizon_gate_mode
        self.propagation_mode = propagation_mode
        self.topology_mode = topology_mode
        self._horizon_calibration_only = False
        self._lag_refinement_only = False
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
            lag_bias_mode,
        )
        self.fusion = LocalUpstreamGatedFusion(hidden_dim)
        self.decoder = MultiHorizonMultiTargetDecoder(hidden_dim, output_window, target_dim)
        self.upstream_decoder = UpstreamResidualDecoder(hidden_dim, target_dim)
        self.horizon_gate = (
            BoundedLinearHorizonGate(output_window)
            if horizon_gate_mode == "linear"
            else None
        )
        self.trajectory_propagation = (
            DirectedTrajectoryPropagation(
                hidden_dim,
                edge_dim,
                max_lag,
                steps=trajectory_steps,
                dropout=dropout,
            )
            if propagation_mode == "trajectory"
            else None
        )
        self.output_transport = (
            DirectedLaggedOutputTransport(
                target_dim,
                edge_dim,
                max_lag,
                steps=trajectory_steps,
                hidden_dim=max(16, hidden_dim // 2),
                dropout=dropout,
            )
            if propagation_mode == "output_transport"
            else None
        )
        self.history_propagation = (
            DirectedLaggedHistoryPropagation(
                hidden_dim,
                edge_dim,
                max_lag,
                steps=trajectory_steps,
                dropout=dropout,
            )
            if propagation_mode == "history"
            else None
        )
        self.topology_encoder = (
            DirectedTopologyEncoder(hidden_dim)
            if topology_mode == "structural"
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
        if self.propagation_mode in {"output_transport", "history"}:
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            residual_module = (
                self.output_transport
                if self.propagation_mode == "output_transport"
                else self.history_propagation
            )
            assert residual_module is not None
            for parameter in residual_module.parameters():
                parameter.requires_grad_(True)
            return
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

    def configure_lag_refinement_training(self) -> None:
        """Freeze edge routing and train only zero-started lag refinements."""
        if self.message_passing.lag_mode != "learned_lag":
            raise ValueError("lag refinement requires lag_mode=learned_lag")
        if self.message_passing.lag_offset_bias is None:
            raise ValueError("lag refinement requires lag_bias_mode=global")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.message_passing.lag_residual_scale.requires_grad_(True)
        self.message_passing.lag_offset_bias.requires_grad_(True)
        self._lag_refinement_only = True

    def train(self, mode: bool = True) -> RiverLagNet:
        """Keep the frozen graph forecaster deterministic during calibration."""
        super().train(mode)
        if mode and (self._horizon_calibration_only or self._lag_refinement_only):
            for module in (
                self.input_encoder,
                self.temporal_encoder,
                self.message_passing,
                self.fusion,
                self.decoder,
                self.upstream_decoder,
            ):
                module.eval()
            if self.horizon_gate is not None:
                self.horizon_gate.eval()
            if self._horizon_calibration_only:
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
        variant_edges: Tensor | None = None
        variant_attr: Tensor | None = None
        if self.graph_variant != "no_graph":
            variant_edges, variant_attr = build_graph_variant(
                edge_index, edge_attr, self.graph_variant, self.graph_seed
            )
        if self.graph_variant != "no_graph" and self.propagation_mode == "history":
            assert self.history_propagation is not None
            assert variant_edges is not None and variant_attr is not None
            encoded, routing = self.history_propagation(
                encoded, variant_edges, variant_attr
            )
            self.attention_weights = routing
        h_seq, h_local = self.temporal_encoder(encoded)
        if self.graph_variant != "no_graph" and self.topology_encoder is not None:
            topology = self.topology_encoder(edge_index, edge_attr, h_local.shape[1])
            h_seq = h_seq + topology[None, None]
            h_local = h_local + topology[None]
        local_context = self.decoder.contextualize(h_local)
        local_prediction = self.decoder.decode_context(local_context)
        if self.graph_variant == "no_graph":
            self.attention_weights = None
            return local_prediction
        else:
            assert variant_edges is not None and variant_attr is not None
            if self.propagation_mode == "history":
                return local_prediction
            if self.propagation_mode == "output_transport":
                assert self.output_transport is not None
                transported, routing = self.output_transport(
                    local_prediction,
                    x[..., : local_prediction.shape[-1]],
                    x_mask[..., : local_prediction.shape[-1]],
                    variant_edges,
                    variant_attr,
                )
                self.attention_weights = routing
                return transported
            if self.propagation_mode == "trajectory":
                assert self.trajectory_propagation is not None
                propagated, routing = self.trajectory_propagation(
                    local_context, h_seq, variant_edges, variant_attr
                )
                self.attention_weights = routing
                upstream_state = propagated - local_context
            else:
                h_upstream, attention = self.message_passing.forward_horizons(
                    h_seq,
                    h_local,
                    variant_edges,
                    variant_attr,
                    output_window=self.output_window,
                )
                self.attention_weights = attention
                fused = self.fusion(local_context, h_upstream)
                upstream_state = fused - local_context
        upstream_correction = self.upstream_decoder(upstream_state)
        if self.horizon_gate is not None:
            upstream_correction = self.horizon_gate(upstream_correction)
        return local_prediction + upstream_correction
