"""Leakage-safe rolling windows over river-network time series."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .normalization import MaskedStandardScaler
from .schema import TARGET_NAMES, TimeSeriesData


class RiverWindowDataset(Dataset[dict[str, Tensor]]):
    """Create past-only inputs and split-contained future targets."""

    def __init__(
        self,
        data: TimeSeriesData,
        scaler: MaskedStandardScaler,
        target_start: int,
        target_end: int,
        input_window: int = 90,
        output_window: int = 30,
    ) -> None:
        if input_window <= 0 or output_window <= 0:
            raise ValueError("input_window and output_window must be positive")
        if target_start < input_window or target_end > data.values.shape[0]:
            raise ValueError("split bounds cannot provide the requested windows")
        self.data = data
        self.scaler = scaler
        self.input_window = input_window
        self.output_window = output_window
        self.target_start = target_start
        self.target_end = target_end
        self.forecast_starts = list(range(target_start, target_end - output_window + 1))
        if not self.forecast_starts:
            raise ValueError("split is shorter than output_window")

    def __len__(self) -> int:
        return len(self.forecast_starts)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        forecast_start = self.forecast_starts[index]
        input_indices = torch.arange(forecast_start - self.input_window, forecast_start)
        target_indices = torch.arange(forecast_start, forecast_start + self.output_window)
        x = self.scaler.transform(self.data.values[input_indices])
        y = self.scaler.transform(self.data.values[target_indices, :, : len(TARGET_NAMES)])
        quality = self.data.quality
        x_quality = (
            quality[input_indices]
            if quality is not None
            else self.data.observed[input_indices].to(self.data.values.dtype)
        )
        return {
            "x": x,
            "x_mask": self.data.observed[input_indices],
            "x_quality": x_quality,
            "time_features": _calendar_features(input_indices),
            "y": y,
            "y_mask": self.data.observed[target_indices, :, : len(TARGET_NAMES)],
            "static": self.data.graph.static,
            "edge_index": self.data.graph.edge_index,
            "edge_attr": self.data.graph.edge_attr,
            "input_indices": input_indices,
            "target_indices": target_indices,
        }

    def all_target_indices(self) -> list[int]:
        """Return sorted target timestamps used anywhere in this split."""
        indices: set[int] = set()
        for start in self.forecast_starts:
            indices.update(range(start, start + self.output_window))
        return sorted(indices)


def river_collate(samples: Sequence[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Batch time-series tensors while retaining a single shared graph."""
    if not samples:
        raise ValueError("cannot collate an empty batch")
    shared = {key: samples[0][key] for key in ("static", "edge_index", "edge_attr")}
    batched = {
        key: torch.stack([sample[key] for sample in samples])
        for key in samples[0]
        if key not in shared
    }
    return {**batched, **shared}


def _calendar_features(indices: Tensor) -> Tensor:
    indices = indices.to(torch.float32)
    annual = 2.0 * math.pi * indices / 365.25
    weekly = 2.0 * math.pi * indices / 7.0
    return torch.stack((annual.sin(), annual.cos(), weekly.sin(), weekly.cos()), dim=-1)
