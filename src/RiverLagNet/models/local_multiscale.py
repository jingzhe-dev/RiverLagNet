"""Strong station-independent multiscale forecasting backbone."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .input_encoder import TargetExogenousInputEncoder


@dataclass
class LocalForecastContext:
    """Intermediate graph-free tensors with every public axis preserved."""

    history_states: Tensor
    scale_states: Tensor
    horizon_states: Tensor
    prediction: Tensor


class _CausalGatedTemporalBlock(nn.Module):
    """Station-shared causal depthwise temporal residual block."""

    def __init__(self, hidden_dim: int, dropout: float, kernel_size: int = 3) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.norm = nn.LayerNorm(hidden_dim)
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            groups=hidden_dim,
        )
        self.gate_projection = nn.Linear(hidden_dim, hidden_dim * 2)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, states: Tensor) -> Tensor:
        if states.ndim != 4:
            raise ValueError("temporal states must have shape [B,T,N,D]")
        batch, history, nodes, hidden = states.shape
        normalized = self.norm(states)
        sequences = normalized.permute(0, 2, 3, 1).reshape(
            batch * nodes, hidden, history
        )
        filtered = self.depthwise(F.pad(sequences, (self.kernel_size - 1, 0)))
        filtered = filtered.reshape(batch, nodes, hidden, history).permute(0, 3, 1, 2)
        values, gates = self.gate_projection(filtered).chunk(2, dim=-1)
        update = self.output_projection(values * torch.sigmoid(gates))
        return states + self.dropout(update)


class LocalMultiscaleForecaster(nn.Module):
    """Forecast each station independently from causal multiscale history."""

    def __init__(
        self,
        value_dim: int,
        static_dim: int,
        time_dim: int,
        edge_dim: int = 0,
        hidden_dim: int = 128,
        output_window: int = 30,
        target_dim: int = 3,
        scales: Sequence[int] = (1, 3, 7, 30),
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        del edge_dim
        normalized_scales = tuple(int(scale) for scale in scales)
        if not normalized_scales or any(scale <= 0 for scale in normalized_scales):
            raise ValueError("scales must contain positive integers")
        if len(set(normalized_scales)) != len(normalized_scales):
            raise ValueError("scales must be unique")
        if normalized_scales[0] != 1:
            raise ValueError("the first scale must preserve the full history")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        self.value_dim = value_dim
        self.hidden_dim = hidden_dim
        self.output_window = output_window
        self.target_dim = target_dim
        self.scales = normalized_scales
        self.input_encoder = TargetExogenousInputEncoder(
            value_dim=value_dim,
            static_dim=static_dim,
            time_dim=time_dim,
            hidden_dim=hidden_dim,
            target_dim=target_dim,
        )
        self.scale_blocks = nn.ModuleList(
            nn.ModuleList(
                _CausalGatedTemporalBlock(hidden_dim, dropout)
                for _ in range(num_layers)
            )
            for _ in self.scales
        )
        self.scale_gate = nn.Sequential(
            nn.LayerNorm(len(self.scales) * hidden_dim),
            nn.Linear(len(self.scales) * hidden_dim, len(self.scales)),
        )
        self.lead_embedding = nn.Parameter(torch.empty(output_window, hidden_dim))
        nn.init.normal_(self.lead_embedding, std=0.02)
        self.horizon_refinement = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.target_heads = nn.ModuleList(
            nn.Linear(hidden_dim, 1) for _ in range(target_dim)
        )

    def encode_context(
        self,
        x: Tensor,
        x_mask: Tensor,
        x_quality: Tensor | None,
        static: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        time_features: Tensor,
    ) -> LocalForecastContext:
        """Encode local history and return all multiscale forecast tensors."""
        del edge_index, edge_attr
        encoded = self.input_encoder(x, x_mask, x_quality, static, time_features)
        observed_time = x_mask.any(dim=-1)
        sequence_states: list[Tensor] = []
        final_states: list[Tensor] = []
        for scale, blocks in zip(self.scales, self.scale_blocks, strict=True):
            states, valid = _masked_pool_history(encoded, observed_time, scale)
            for block in blocks:
                states = block(states)
                states = torch.where(valid[..., None], states, torch.zeros_like(states))
            sequence_states.append(states)
            final_states.append(_last_valid_state(states, valid))

        history_states = sequence_states[0]
        scale_states = torch.stack(final_states, dim=1)
        batch, _, nodes, hidden = scale_states.shape
        gate_input = scale_states.permute(0, 2, 1, 3).reshape(batch, nodes, -1)
        scale_weights = torch.softmax(self.scale_gate(gate_input), dim=-1)
        fused = (
            scale_states * scale_weights.permute(0, 2, 1).unsqueeze(-1)
        ).sum(dim=1)
        horizon_states = fused[:, None] + self.lead_embedding[None, :, None]
        horizon_states = horizon_states + self.horizon_refinement(horizon_states)
        prediction = torch.cat(
            [head(horizon_states) for head in self.target_heads], dim=-1
        )
        if prediction.shape != (
            batch,
            self.output_window,
            nodes,
            self.target_dim,
        ):
            raise RuntimeError("local multiscale decoder violated its public shape")
        return LocalForecastContext(
            history_states=history_states,
            scale_states=scale_states,
            horizon_states=horizon_states,
            prediction=prediction,
        )

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
        """Return direct forecasts with shape ``[B,T_out,N,3]``."""
        return self.encode_context(
            x,
            x_mask,
            x_quality,
            static,
            edge_index,
            edge_attr,
            time_features,
        ).prediction


def _masked_pool_history(
    states: Tensor, observed_time: Tensor, scale: int
) -> tuple[Tensor, Tensor]:
    """Pool trailing causal blocks using only timestamps with observations."""
    batch, history, nodes, hidden = states.shape
    if observed_time.shape != (batch, history, nodes):
        raise ValueError("observed_time must have shape [B,T,N]")
    left_padding = (-history) % scale
    if left_padding:
        states = torch.cat(
            (states.new_zeros(batch, left_padding, nodes, hidden), states), dim=1
        )
        observed_time = torch.cat(
            (
                torch.zeros(
                    batch,
                    left_padding,
                    nodes,
                    dtype=torch.bool,
                    device=observed_time.device,
                ),
                observed_time,
            ),
            dim=1,
        )
    blocks = states.shape[1] // scale
    grouped_states = states.reshape(batch, blocks, scale, nodes, hidden)
    grouped_mask = observed_time.reshape(batch, blocks, scale, nodes)
    counts = grouped_mask.sum(dim=2)
    pooled = (
        grouped_states * grouped_mask[..., None].to(states.dtype)
    ).sum(dim=2) / counts.clamp_min(1)[..., None]
    valid = counts > 0
    pooled = torch.where(valid[..., None], pooled, torch.zeros_like(pooled))
    return pooled, valid


def _last_valid_state(states: Tensor, valid: Tensor) -> Tensor:
    """Select each station's final valid temporal state without node mixing."""
    batch, history, nodes, hidden = states.shape
    time_index = torch.arange(history, device=states.device).view(1, history, 1)
    last_index = torch.where(valid, time_index, -1).amax(dim=1)
    gather_index = last_index.clamp_min(0).view(batch, 1, nodes, 1).expand(
        -1, 1, -1, hidden
    )
    selected = states.gather(dim=1, index=gather_index).squeeze(1)
    return torch.where(last_index[..., None] >= 0, selected, torch.zeros_like(selected))
