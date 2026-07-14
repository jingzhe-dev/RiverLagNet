"""Target-conditioned fusion of upstream water-quality forecast trajectories."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TargetConditionedForecastTransport(nn.Module):
    """Convert path-lag attention into an explicit target-space correction.

    Routing heads are mixed separately for each water-quality target. Source
    candidates use observed history when available and the local Transformer
    forecast after the forecast origin. A zero-initialized residual network
    makes the complete graph correction exactly zero at initialization.
    """

    def __init__(
        self,
        hidden_dim: int,
        target_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or target_dim <= 0 or num_heads <= 0:
            raise ValueError("hidden_dim, target_dim, and num_heads must be positive")
        self.hidden_dim = hidden_dim
        self.target_dim = target_dim
        self.num_heads = num_heads
        self.target_head_logits = nn.Parameter(torch.zeros(target_dim, num_heads))
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim * 2 + target_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, target_dim, bias=False),
        )
        nn.init.zeros_(self.residual[-1].weight)
        self.target_head_weights: Tensor | None = None
        self.upstream_forecast: Tensor | None = None

    def forward(
        self,
        history_targets: Tensor,
        history_mask: Tensor,
        local_prediction: Tensor,
        local_context: Tensor,
        graph_context: Tensor,
        edge_index: Tensor,
        attention_weights: Tensor,
    ) -> Tensor:
        """Return target-space correction ``[B,H,N,C]``."""
        self._validate(
            history_targets,
            history_mask,
            local_prediction,
            local_context,
            graph_context,
            edge_index,
            attention_weights,
        )
        batch, horizons, nodes, targets = local_prediction.shape
        candidates, candidate_valid = self._candidate_targets(
            history_targets,
            history_mask,
            local_prediction,
            edge_index,
            attention_weights.shape[3],
        )
        self.target_head_weights = torch.softmax(
            self.target_head_logits, dim=-1
        ).to(attention_weights.dtype)
        target_weights = torch.einsum(
            "bhelr,cr->bhelc", attention_weights, self.target_head_weights
        )
        target_weights = target_weights * candidate_valid.to(target_weights.dtype)
        edge_denominator = target_weights.sum(dim=3)
        edge_numerator = (
            target_weights * candidates.to(target_weights.dtype)
        ).sum(dim=3)
        destination = edge_index[1]
        denominator = edge_denominator.new_zeros(batch, horizons, nodes, targets)
        numerator = edge_numerator.new_zeros(batch, horizons, nodes, targets)
        denominator.index_add_(2, destination, edge_denominator)
        numerator.index_add_(2, destination, edge_numerator)
        valid = denominator > 0
        upstream = (numerator / denominator.clamp_min(1e-8)).to(
            local_prediction.dtype
        )
        upstream = upstream * valid.to(upstream.dtype)
        self.upstream_forecast = upstream
        features = torch.cat(
            (
                local_context,
                graph_context,
                local_prediction,
                upstream,
                upstream - local_prediction,
            ),
            dim=-1,
        )
        correction = self.residual(features)
        return correction * valid.to(correction.dtype)

    def _candidate_targets(
        self,
        history_targets: Tensor,
        history_mask: Tensor,
        local_prediction: Tensor,
        edge_index: Tensor,
        lag_count: int,
    ) -> tuple[Tensor, Tensor]:
        """Build leakage-free observed/forecast target candidates."""
        _, history, _, _ = history_targets.shape
        horizons = local_prediction.shape[1]
        source = edge_index[0]
        lead = torch.arange(1, horizons + 1, device=history_targets.device)
        lag = torch.arange(lag_count, device=history_targets.device)
        relative_lead = lead[:, None] - lag[None, :]
        history_time = history - 1 + relative_lead
        feasible_history = history_time >= 0
        history_time = history_time.clamp(0, history - 1)
        future_time = (relative_lead - 1).clamp(0, horizons - 1)
        history_index = history_time[:, None, :]
        future_index = future_time[:, None, :]
        source_index = source[None, :, None]
        observed_candidates = history_targets[:, history_index, source_index]
        observed_valid = history_mask[:, history_index, source_index]
        forecast_candidates = local_prediction[:, future_index, source_index]
        use_forecast = relative_lead > 0
        candidates = torch.where(
            use_forecast[None, :, None, :, None],
            forecast_candidates,
            observed_candidates,
        ).to(local_prediction.dtype)
        candidate_valid = torch.where(
            use_forecast[None, :, None, :, None],
            torch.ones_like(observed_valid),
            observed_valid & feasible_history[None, :, None, :, None],
        )
        return candidates, candidate_valid

    def _validate(
        self,
        history_targets: Tensor,
        history_mask: Tensor,
        local_prediction: Tensor,
        local_context: Tensor,
        graph_context: Tensor,
        edge_index: Tensor,
        attention_weights: Tensor,
    ) -> None:
        if history_targets.ndim != 4 or history_targets.shape[-1] != self.target_dim:
            raise ValueError("history_targets must have shape [B,T,N,target_dim]")
        if history_mask.shape != history_targets.shape or history_mask.dtype != torch.bool:
            raise ValueError("history_mask must be a bool tensor matching history_targets")
        if local_prediction.ndim != 4 or local_prediction.shape[-1] != self.target_dim:
            raise ValueError("local_prediction must have shape [B,H,N,target_dim]")
        expected_context = (*local_prediction.shape[:-1], self.hidden_dim)
        if local_context.shape != expected_context or graph_context.shape != expected_context:
            raise ValueError(f"local and graph contexts must have shape {expected_context}")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2,E]")
        if attention_weights.ndim != 5:
            raise ValueError("attention_weights must have shape [B,H,E,L,R]")
        expected_attention = (
            local_prediction.shape[0],
            local_prediction.shape[1],
            edge_index.shape[1],
        )
        if attention_weights.shape[:3] != expected_attention:
            raise ValueError("attention weights do not match batch, horizon, or edges")
        if attention_weights.shape[-1] != self.num_heads:
            raise ValueError("attention weights have the wrong routing-head dimension")
