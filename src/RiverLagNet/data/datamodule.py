"""Lightning data module for chronological synthetic river experiments."""

from __future__ import annotations

from dataclasses import dataclass

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from .dataset import RiverWindowDataset, river_collate
from .normalization import MaskedStandardScaler
from .schema import TimeSeriesData
from .synthetic import generate_synthetic_river_data


@dataclass(frozen=True)
class DataSpec:
    """Dimensions needed to construct forecasting models."""

    num_nodes: int
    num_variables: int
    static_dim: int
    edge_dim: int
    time_dim: int = 4


class RiverDataModule(LightningDataModule):
    """Create deterministic 70/15/15 chronological river data splits."""

    def __init__(
        self,
        num_days: int = 260,
        num_nodes: int = 8,
        num_variables: int = 3,
        missing_rate: float = 0.08,
        input_window: int = 90,
        output_window: int = 30,
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.data: TimeSeriesData | None = None
        self.scaler: MaskedStandardScaler | None = None
        self.train_dataset: RiverWindowDataset
        self.val_dataset: RiverWindowDataset
        self.test_dataset: RiverWindowDataset
        self.train_end = int(num_days * 0.70)
        self.val_end = int(num_days * 0.85)

    def setup(self, stage: str | None = None) -> None:
        """Generate data once, fit train-only statistics, and build split windows."""
        if self.data is None:
            self.data = generate_synthetic_river_data(
                num_days=self.hparams.num_days,
                num_nodes=self.hparams.num_nodes,
                num_variables=self.hparams.num_variables,
                missing_rate=self.hparams.missing_rate,
                seed=self.hparams.seed,
            )
            self.scaler = MaskedStandardScaler().fit(
                self.data.values[: self.train_end], self.data.observed[: self.train_end]
            )
        assert self.scaler is not None
        self.train_dataset = RiverWindowDataset(
            self.data,
            self.scaler,
            self.hparams.input_window,
            self.train_end,
            self.hparams.input_window,
            self.hparams.output_window,
        )
        self.val_dataset = RiverWindowDataset(
            self.data,
            self.scaler,
            self.train_end,
            self.val_end,
            self.hparams.input_window,
            self.hparams.output_window,
        )
        self.test_dataset = RiverWindowDataset(
            self.data,
            self.scaler,
            self.val_end,
            self.hparams.num_days,
            self.hparams.input_window,
            self.hparams.output_window,
        )

    @property
    def data_spec(self) -> DataSpec:
        """Return feature dimensions after setup."""
        if self.data is None:
            raise RuntimeError("setup must be called before requesting data_spec")
        return DataSpec(
            num_nodes=self.data.values.shape[1],
            num_variables=self.data.values.shape[2],
            static_dim=self.data.graph.static.shape[1],
            edge_dim=self.data.graph.edge_attr.shape[1],
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)

    def _loader(self, dataset: RiverWindowDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            collate_fn=river_collate,
            persistent_workers=self.hparams.num_workers > 0,
            pin_memory=self.hparams.pin_memory,
        )
