"""Persistence, station-only GRU, and static directed GAT baselines."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import GATConv

from .decoder import MultiHorizonMultiTargetDecoder
from .input_encoder import InputMaskEncoder
from .temporal_gru import NodeTemporalGRU


class PersistenceModel(nn.Module):
    """Repeat each station's most recent observed target value."""

    def __init__(self, output_window: int = 30, target_dim: int = 3, **_: object) -> None:
        super().__init__()
        self.output_window = output_window
        self.target_dim = target_dim

    def forward(
        self,
        x: Tensor,
        x_mask: Tensor,
        x_quality: Tensor | None = None,
        static: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_attr: Tensor | None = None,
        time_features: Tensor | None = None,
    ) -> Tensor:
        """Return `[B,T_out,N,target_dim]` persistence forecasts."""
        target_values = x[..., : self.target_dim]
        target_mask = x_mask[..., : self.target_dim]
        history = x.shape[1]
        time_ids = torch.arange(history, device=x.device).view(1, history, 1, 1)
        last_ids = torch.where(target_mask, time_ids, -1).amax(dim=1)
        gather_ids = last_ids.clamp_min(0).unsqueeze(1)
        last = target_values.gather(1, gather_ids).squeeze(1)
        last = torch.where(last_ids >= 0, last, torch.zeros_like(last))
        return last[:, None].expand(-1, self.output_window, -1, -1)


class StationGRU(nn.Module):
    """Station-independent GRU baseline without graph information."""

    def __init__(
        self,
        value_dim: int,
        static_dim: int,
        time_dim: int,
        hidden_dim: int = 64,
        output_window: int = 30,
        target_dim: int = 3,
        edge_dim: int = 0,
        **_: object,
    ) -> None:
        super().__init__()
        self.input_encoder = InputMaskEncoder(value_dim, static_dim, time_dim, hidden_dim)
        self.temporal_encoder = NodeTemporalGRU(hidden_dim, hidden_dim)
        self.decoder = MultiHorizonMultiTargetDecoder(hidden_dim, output_window, target_dim)

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
        encoded = self.input_encoder(x, x_mask, x_quality, static, time_features)
        _, local = self.temporal_encoder(encoded)
        return self.decoder(local)


class StaticDirectedGAT(StationGRU):
    """Directed GAT baseline using only each station's latest GRU state."""

    def __init__(self, edge_dim: int, hidden_dim: int = 64, **kwargs: object) -> None:
        super().__init__(edge_dim=edge_dim, hidden_dim=hidden_dim, **kwargs)
        self.gat = GATConv(
            hidden_dim,
            hidden_dim,
            heads=1,
            concat=False,
            edge_dim=edge_dim,
            add_self_loops=True,
        )
        self.gat_norm = nn.LayerNorm(hidden_dim)

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
        encoded = self.input_encoder(x, x_mask, x_quality, static, time_features)
        _, local = self.temporal_encoder(encoded)
        graph_states = torch.stack(
            [self.gat(sample, edge_index, edge_attr) for sample in local], dim=0
        )
        return self.decoder(self.gat_norm(local + graph_states))
