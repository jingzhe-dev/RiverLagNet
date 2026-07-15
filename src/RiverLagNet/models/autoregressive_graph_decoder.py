"""Causal autoregressive decoder with directed edge-lag graph transitions."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .graph_cross_attention import sparsemax


class RecursiveCausalEdgeLagAttention(nn.Module):
    """Route observed or already-predicted upstream states into one future step.

    Candidate lag ``tau`` is restricted to ``1..max_lag``. At forecast lead
    ``h``, ``h - tau <= 0`` selects an observed history state and
    ``h - tau > 0`` selects an earlier model-predicted state. Consequently the
    attention can propagate information through the forecast trajectory but
    can never read the current or a future target.

    Attention is normalized jointly over every incoming path-lag candidate for
    each destination and head. A query-conditioned shift adapts the physical
    travel-time prior to the current Transformer state without changing edge
    direction.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        num_heads: int = 4,
        max_lag: int = 30,
        max_path_hops: int = 8,
        prior_scale_days: float = 2.0,
        max_dynamic_shift_days: float = 2.0,
        value_mode: str = "state",
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if edge_dim <= 0 or max_lag <= 0 or max_path_hops <= 0:
            raise ValueError("edge_dim, max_lag, and max_path_hops must be positive")
        if prior_scale_days <= 0 or max_dynamic_shift_days < 0:
            raise ValueError("prior scale must be positive and dynamic shift non-negative")
        if value_mode not in {"state", "innovation", "adaptive"}:
            raise ValueError("value_mode must be state, innovation, or adaptive")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.max_lag = max_lag
        self.max_path_hops = max_path_hops
        self.max_dynamic_shift_days = max_dynamic_shift_days
        self.value_mode = value_mode

        self.query_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.innovation_projection = (
            nn.Parameter(torch.zeros(hidden_dim, hidden_dim))
            if value_mode == "adaptive"
            else None
        )
        self.edge_key = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.edge_bias = nn.Linear(edge_dim, num_heads, bias=False)
        self.lag_shift = nn.Linear(hidden_dim, num_heads, bias=False)
        self.lag_embedding = nn.Parameter(
            torch.empty(max_lag, num_heads, self.head_dim)
        )
        self.path_embedding = nn.Parameter(
            torch.empty(max_path_hops + 1, num_heads, self.head_dim)
        )
        nn.init.normal_(self.lag_embedding, std=0.02)
        nn.init.normal_(self.path_embedding, std=0.02)
        inverse_softplus = math.log(math.exp(prior_scale_days) - 1.0)
        self.prior_raw_scale = nn.Parameter(
            torch.full((num_heads,), inverse_softplus)
        )

    def forward(
        self,
        history_states: Tensor,
        future_states: Tensor,
        destination_query: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_hops: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return contexts ``[B,N,R,d]`` and weights ``[B,E,L,R]``."""
        self._validate(
            history_states,
            future_states,
            destination_query,
            edge_index,
            edge_attr,
        )
        batch, _, nodes, _ = history_states.shape
        edges = edge_index.shape[1]
        if edges == 0:
            context = history_states.new_zeros(
                batch, nodes, self.num_heads, self.head_dim
            )
            weights = history_states.new_zeros(
                batch, 0, self.max_lag, self.num_heads
            )
            return context, weights

        if edge_hops is None:
            edge_hops = torch.ones(
                edges, dtype=torch.long, device=edge_index.device
            )
        if edge_hops.shape != (edges,) or edge_hops.dtype != torch.long:
            raise ValueError("edge_hops must have shape [E] and torch.long dtype")

        source, destination = edge_index
        candidate_states, feasible = self._candidate_states(
            history_states, future_states, source
        )
        candidate_keys = self.key_projection(candidate_states).view(
            batch,
            edges,
            self.max_lag,
            self.num_heads,
            self.head_dim,
        )
        projected_values = self._project_message_values(
            candidate_states, destination_query, destination
        )
        candidate_values = projected_values.view(
            batch,
            edges,
            self.max_lag,
            self.num_heads,
            self.head_dim,
        )
        queries = self.query_projection(destination_query).view(
            batch, nodes, self.num_heads, self.head_dim
        )
        edge_keys = self.edge_key(edge_attr).view(
            edges, self.num_heads, self.head_dim
        )
        path_keys = self.path_embedding[
            edge_hops.clamp(1, self.max_path_hops)
        ]
        candidate_keys = (
            candidate_keys
            + edge_keys[None, :, None]
            + path_keys[None, :, None]
            + self.lag_embedding[None, None]
        )
        scores = (
            queries[:, destination, None] * candidate_keys
        ).sum(dim=-1) / math.sqrt(self.head_dim)
        scores = scores + self.edge_bias(edge_attr)[None, :, None]

        query_shift = self.max_dynamic_shift_days * torch.tanh(
            self.lag_shift(destination_query)
        )
        travel_time = edge_attr[:, -1].to(scores.dtype)
        center = travel_time[None, :, None] + query_shift[:, destination]
        lag_days = torch.arange(
            1,
            self.max_lag + 1,
            device=scores.device,
            dtype=scores.dtype,
        )
        prior_scale = F.softplus(self.prior_raw_scale).to(scores.dtype) + 0.25
        prior = -0.5 * (
            (lag_days[None, None, :, None] - center[:, :, None])
            / prior_scale[None, None, None]
        ).square()
        scores = scores + prior
        scores = scores.masked_fill(
            ~feasible[None, None, :, None], -1e4
        )
        return self._normalize_and_aggregate(
            scores, candidate_values, destination, nodes
        )

    def _message_states(
        self,
        candidate_states: Tensor,
        destination_query: Tensor,
        destination: Tensor,
    ) -> Tensor:
        """Return absolute upstream states or destination-relative innovations."""
        if self.value_mode == "state":
            return candidate_states
        return candidate_states - destination_query[:, destination, None]

    def _project_message_values(
        self,
        candidate_states: Tensor,
        destination_query: Tensor,
        destination: Tensor,
    ) -> Tensor:
        """Project absolute state and an optional zero-start innovation channel."""
        if self.value_mode != "adaptive":
            return self.value_projection(
                self._message_states(
                    candidate_states, destination_query, destination
                )
            )
        innovation = (
            candidate_states - destination_query[:, destination, None]
        )
        if self.innovation_projection is None:
            raise RuntimeError("adaptive value mode requires an innovation projection")
        return self.value_projection(candidate_states) + F.linear(
            innovation, self.innovation_projection
        )

    def _candidate_states(
        self,
        history_states: Tensor,
        future_states: Tensor,
        source: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Build candidates without current-step or future-state leakage."""
        _, history, _, hidden = history_states.shape
        previous_leads = future_states.shape[1]
        lead = previous_leads + 1
        zero = history_states.new_zeros(
            history_states.shape[0], source.numel(), hidden
        )
        candidates: list[Tensor] = []
        feasible: list[bool] = []
        for lag in range(1, self.max_lag + 1):
            relative_lead = lead - lag
            if relative_lead <= 0:
                history_index = history - 1 + relative_lead
                is_feasible = history_index >= 0
                candidate = (
                    history_states[:, history_index, source]
                    if is_feasible
                    else zero
                )
            else:
                future_index = relative_lead - 1
                if future_index >= previous_leads:
                    raise RuntimeError("attention attempted to read an unavailable future state")
                is_feasible = True
                candidate = future_states[:, future_index, source]
            candidates.append(candidate)
            feasible.append(is_feasible)
        return (
            torch.stack(candidates, dim=2),
            torch.tensor(feasible, dtype=torch.bool, device=history_states.device),
        )

    def _normalize_and_aggregate(
        self,
        scores: Tensor,
        candidate_values: Tensor,
        destination: Tensor,
        nodes: int,
    ) -> tuple[Tensor, Tensor]:
        """Jointly normalize incoming edge-lag candidates per node and head."""
        batch, edges, lags, heads = scores.shape
        weights = torch.zeros_like(scores)
        contexts = candidate_values.new_zeros(
            batch, nodes, heads, self.head_dim
        )
        incoming_count = torch.bincount(destination, minlength=nodes)
        max_incoming = int(incoming_count.max())
        if max_incoming <= 1:
            normalized = sparsemax(
                scores.permute(0, 1, 3, 2).float(), dim=-1
            ).to(scores.dtype).permute(0, 1, 3, 2)
            weights = normalized
            contexts[:, destination] = (
                normalized.to(candidate_values.dtype)[..., None]
                * candidate_values
            ).sum(dim=2).to(contexts.dtype)
            return contexts, weights

        padded_edges = torch.zeros(
            nodes, max_incoming, dtype=torch.long, device=destination.device
        )
        padded_valid = torch.zeros(
            nodes, max_incoming, dtype=torch.bool, device=destination.device
        )
        node_edge_ids: list[Tensor] = []
        for node in range(nodes):
            edge_ids = torch.nonzero(destination == node, as_tuple=False).flatten()
            node_edge_ids.append(edge_ids)
            if edge_ids.numel():
                padded_edges[node, : edge_ids.numel()] = edge_ids
                padded_valid[node, : edge_ids.numel()] = True

        padded_scores = scores[:, padded_edges]
        padded_scores = padded_scores.masked_fill(
            ~padded_valid[None, :, :, None, None], -1e4
        )
        flattened = padded_scores.permute(0, 1, 4, 2, 3).flatten(3)
        padded_weights = sparsemax(flattened.float(), dim=-1).to(
            scores.dtype
        ).reshape(batch, nodes, heads, max_incoming, lags).permute(
            0, 1, 3, 4, 2
        )
        padded_weights = padded_weights * padded_valid[
            None, :, :, None, None
        ].to(padded_weights.dtype)
        padded_weights = padded_weights / padded_weights.sum(
            dim=(2, 3), keepdim=True
        ).clamp_min(1e-8)
        padded_values = candidate_values[:, padded_edges]
        contexts = (
            padded_weights.to(padded_values.dtype)[..., None]
            * padded_values
        ).sum(dim=(2, 3)).to(padded_values.dtype)
        for node, edge_ids in enumerate(node_edge_ids):
            if edge_ids.numel():
                weights[:, edge_ids] = padded_weights[
                    :, node, : edge_ids.numel()
                ]
        return contexts, weights

    def _validate(
        self,
        history_states: Tensor,
        future_states: Tensor,
        destination_query: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> None:
        if history_states.ndim != 4:
            raise ValueError("history_states must have shape [B,T,N,D]")
        if future_states.ndim != 4:
            raise ValueError("future_states must have shape [B,H,N,D]")
        if destination_query.ndim != 3:
            raise ValueError("destination_query must have shape [B,N,D]")
        batch, _, nodes, hidden = history_states.shape
        if hidden != self.hidden_dim:
            raise ValueError("history hidden dimension mismatch")
        if future_states.shape[0] != batch or future_states.shape[2:] != (
            nodes,
            hidden,
        ):
            raise ValueError("future state shape mismatch")
        if destination_query.shape != (batch, nodes, hidden):
            raise ValueError("destination query shape mismatch")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2,E]")
        if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_attr must have shape [E,A]")


class GraphModulatedRecurrentFusion(nn.Module):
    """Inject GNN messages inside a Transformer-state transition.

    Head competition selects complementary graph subspaces. The selected GNN
    message then produces a gated affine modulation of the local Transformer
    state. The final projection is zero-initialized, making the directed model
    exactly equal to its no-graph counterpart before graph learning starts.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        if hidden_dim <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.query_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.message_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.modulation = nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)
        self.gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.output_projection.weight)

    def forward(
        self, local_state: Tensor, head_contexts: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return fused state ``[B,N,D]`` and head weights ``[B,N,R]``."""
        if local_state.ndim != 3:
            raise ValueError("local_state must have shape [B,N,D]")
        expected = (*local_state.shape[:2], self.num_heads, self.head_dim)
        if head_contexts.shape != expected:
            raise ValueError("head_contexts must have shape [B,N,R,d]")
        query = self.query_projection(local_state).view(
            *local_state.shape[:2], self.num_heads, self.head_dim
        )
        head_scores = (query * head_contexts).sum(dim=-1) / math.sqrt(
            self.head_dim
        )
        head_weights = torch.softmax(head_scores.float(), dim=-1).to(
            local_state.dtype
        )
        weighted_heads = (
            head_contexts * head_weights[..., None]
        ).reshape(*local_state.shape[:2], self.hidden_dim)
        message = self.message_projection(weighted_heads)
        scale, shift = self.modulation(message).chunk(2, dim=-1)
        carrier = torch.tanh(scale) * local_state + shift
        gate = torch.sigmoid(self.gate(torch.cat([local_state, message], dim=-1)))
        delta = self.output_projection(gate * F.silu(carrier))
        return local_state + delta, head_weights


class DirectedAutoregressiveGraphDecoder(nn.Module):
    """Decode future water quality through causal graph-state recurrence."""

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        output_window: int = 30,
        target_dim: int = 3,
        num_heads: int = 4,
        max_lag: int = 30,
        max_path_hops: int = 8,
        dropout: float = 0.1,
        prior_scale_days: float = 2.0,
        max_dynamic_shift_days: float = 2.0,
        attention_value_mode: str = "state",
        counterfactual_output_fusion: bool = False,
    ) -> None:
        super().__init__()
        if output_window <= 0 or target_dim <= 0:
            raise ValueError("output_window and target_dim must be positive")
        self.hidden_dim = hidden_dim
        self.output_window = output_window
        self.target_dim = target_dim
        self.num_heads = num_heads
        self.max_lag = max_lag
        self.counterfactual_output_fusion = counterfactual_output_fusion
        self.horizon_embedding = nn.Parameter(
            torch.empty(output_window, hidden_dim)
        )
        nn.init.normal_(self.horizon_embedding, std=0.02)
        self.feedback_encoder = nn.Linear(2 * target_dim, hidden_dim, bias=False)
        self.local_transition = nn.GRUCell(hidden_dim, hidden_dim)
        self.local_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = RecursiveCausalEdgeLagAttention(
            hidden_dim,
            edge_dim,
            num_heads=num_heads,
            max_lag=max_lag,
            max_path_hops=max_path_hops,
            prior_scale_days=prior_scale_days,
            max_dynamic_shift_days=max_dynamic_shift_days,
            value_mode=attention_value_mode,
        )
        self.fusion = GraphModulatedRecurrentFusion(hidden_dim, num_heads)
        self.output_shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.output_heads = nn.ModuleList(
            nn.Linear(hidden_dim, 1) for _ in range(target_dim)
        )
        self.counterfactual_gate_logits = (
            nn.Parameter(torch.zeros(output_window, target_dim))
            if counterfactual_output_fusion
            else None
        )
        self.output_gate_values: Tensor | None = None

    def forward(
        self,
        history_states: Tensor,
        initial_state: Tensor,
        previous_values: Tensor,
        previous_mask: Tensor,
        edge_index: Tensor | None = None,
        edge_attr: Tensor | None = None,
        edge_hops: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Return predictions and optional edge-lag/head routing tensors."""
        self._validate_initial_inputs(
            history_states, initial_state, previous_values, previous_mask
        )
        use_graph = edge_index is not None and edge_index.shape[1] > 0
        if use_graph and edge_attr is None:
            raise ValueError("edge_attr is required when graph edges are provided")

        batch, _, nodes, _ = history_states.shape
        state = initial_state
        local_state = initial_state
        future_states: list[Tensor] = []
        predictions: list[Tensor] = []
        routing: list[Tensor] = []
        fusion_weights: list[Tensor] = []
        values = previous_values
        local_values = previous_values
        mask = previous_mask.to(previous_values.dtype)
        local_mask = mask
        for horizon in range(self.output_window):
            feedback = self.feedback_encoder(torch.cat([values, mask], dim=-1))
            transition_input = feedback + self.horizon_embedding[horizon][None, None]
            use_counterfactual = self.counterfactual_output_fusion and use_graph
            if use_counterfactual:
                local_feedback = self.feedback_encoder(
                    torch.cat([local_values, local_mask], dim=-1)
                )
                local_transition_input = (
                    local_feedback + self.horizon_embedding[horizon][None, None]
                )
                transition_input, local_transition_input = self._paired_dropout(
                    transition_input, local_transition_input
                )
                local_state = self.local_transition(
                    local_transition_input.reshape(
                        batch * nodes, self.hidden_dim
                    ),
                    local_state.reshape(batch * nodes, self.hidden_dim),
                ).reshape(batch, nodes, self.hidden_dim)
                local_state = self.local_norm(local_state)
            else:
                transition_input = self.dropout(transition_input)
            state = self.local_transition(
                transition_input.reshape(batch * nodes, self.hidden_dim),
                state.reshape(batch * nodes, self.hidden_dim),
            ).reshape(batch, nodes, self.hidden_dim)
            state = self.local_norm(state)

            if use_graph:
                assert edge_index is not None and edge_attr is not None
                previous_states = (
                    torch.stack(future_states, dim=1)
                    if future_states
                    else history_states.new_empty(
                        batch, 0, nodes, self.hidden_dim
                    )
                )
                head_contexts, step_routing = self.attention(
                    history_states,
                    previous_states,
                    state,
                    edge_index,
                    edge_attr,
                    edge_hops,
                )
                state, step_fusion = self.fusion(state, head_contexts)
                routing.append(step_routing)
                fusion_weights.append(step_fusion)

            decoded = self.output_shared(state)
            graph_prediction = torch.cat(
                [head(decoded) for head in self.output_heads], dim=-1
            )
            if use_counterfactual:
                local_decoded = self.output_shared(local_state)
                local_prediction = torch.cat(
                    [head(local_decoded) for head in self.output_heads], dim=-1
                )
                if self.counterfactual_gate_logits is None:
                    raise RuntimeError("counterfactual fusion requires gate logits")
                output_gate = torch.sigmoid(
                    self.counterfactual_gate_logits[horizon]
                ).to(graph_prediction.dtype)
                prediction = local_prediction + output_gate[None, None] * (
                    graph_prediction - local_prediction
                )
                local_values = local_prediction
                local_mask = torch.ones_like(local_prediction)
            else:
                prediction = graph_prediction
            predictions.append(prediction)
            future_states.append(state)
            values = prediction
            mask = torch.ones_like(prediction)

        output = torch.stack(predictions, dim=1)
        if not use_graph:
            self.output_gate_values = None
            return output, None, None
        self.output_gate_values = (
            torch.sigmoid(self.counterfactual_gate_logits)
            if self.counterfactual_gate_logits is not None
            else None
        )
        return (
            output,
            torch.stack(routing, dim=1),
            torch.stack(fusion_weights, dim=1),
        )

    def _paired_dropout(
        self, graph_input: Tensor, local_input: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Apply one dropout mask to paired graph and counterfactual inputs."""
        if not self.training or self.dropout.p == 0.0:
            return graph_input, local_input
        keep_probability = 1.0 - self.dropout.p
        multiplier = torch.empty_like(graph_input).bernoulli_(keep_probability)
        multiplier = multiplier / keep_probability
        return graph_input * multiplier, local_input * multiplier

    def _validate_initial_inputs(
        self,
        history_states: Tensor,
        initial_state: Tensor,
        previous_values: Tensor,
        previous_mask: Tensor,
    ) -> None:
        if history_states.ndim != 4:
            raise ValueError("history_states must have shape [B,T,N,D]")
        batch, _, nodes, hidden = history_states.shape
        if hidden != self.hidden_dim:
            raise ValueError("history hidden dimension mismatch")
        if initial_state.shape != (batch, nodes, hidden):
            raise ValueError("initial_state must have shape [B,N,D]")
        expected_target = (batch, nodes, self.target_dim)
        if previous_values.shape != expected_target:
            raise ValueError("previous_values must have shape [B,N,C]")
        if previous_mask.shape != expected_target:
            raise ValueError("previous_mask must have shape [B,N,C]")
