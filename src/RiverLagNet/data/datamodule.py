"""Lightning data module for chronological synthetic river experiments."""

from __future__ import annotations

from dataclasses import dataclass

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from .dataset import RiverWindowDataset, river_collate
from .normalization import MaskedStandardScaler
from .real_daily import load_real_daily_dataset
from .schema import TimeSeriesData
from .splits import select_v02_fold
from .synthetic import generate_synthetic_river_data
from .synthetic_identifiable import (
    SyntheticScenario,
    generate_identifiable_synthetic_scenario,
)


@dataclass(frozen=True)
class DataSpec:
    """Dimensions needed to construct forecasting models."""

    num_nodes: int
    num_variables: int
    static_dim: int
    edge_dim: int
    time_dim: int = 4


class RiverDataModule(LightningDataModule):
    """Create deterministic legacy or sealed v0.2 chronological splits."""

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
        persistent_workers: bool | None = None,
        prefetch_factor: int | None = 2,
        seed: int = 42,
        scenario: str = "legacy",
        dataset_path: str | None = None,
        split_name: str = "legacy",
    ) -> None:
        super().__init__()
        if scenario not in {"legacy", "identifiable_v1", "real_daily"}:
            raise ValueError("scenario must be legacy, identifiable_v1, or real_daily")
        if scenario == "real_daily" and not dataset_path:
            raise ValueError("dataset_path is required for scenario=real_daily")
        self.save_hyperparameters()
        self.data: TimeSeriesData | None = None
        self.synthetic_scenario: SyntheticScenario | None = None
        self.scaler: MaskedStandardScaler | None = None
        self.train_dataset: RiverWindowDataset
        self.val_dataset: RiverWindowDataset
        self.test_dataset: RiverWindowDataset | None = None
        self.train_end = int(num_days * 0.70)
        self.val_end = int(num_days * 0.85)
        self.final_test_start = self.val_end

    def setup(self, stage: str | None = None) -> None:
        """Generate data once, fit train-only statistics, and build split windows."""
        if self.data is None:
            if self.hparams.scenario == "real_daily":
                self.data = load_real_daily_dataset(self.hparams.dataset_path)
            elif self.hparams.scenario == "identifiable_v1":
                self.synthetic_scenario = generate_identifiable_synthetic_scenario(
                    num_days=self.hparams.num_days,
                    num_nodes=self.hparams.num_nodes,
                    num_variables=self.hparams.num_variables,
                    missing_rate=self.hparams.missing_rate,
                    seed=self.hparams.seed,
                )
                self.data = self.synthetic_scenario.data
            else:
                self.data = generate_synthetic_river_data(
                    num_days=self.hparams.num_days,
                    num_nodes=self.hparams.num_nodes,
                    num_variables=self.hparams.num_variables,
                    missing_rate=self.hparams.missing_rate,
                    seed=self.hparams.seed,
                )
            num_days = self.data.values.shape[0]
            if self.hparams.split_name == "legacy":
                self.train_end = int(num_days * 0.70)
                self.val_end = int(num_days * 0.85)
                self.final_test_start = self.val_end
            else:
                fold = select_v02_fold(self.hparams.split_name, num_days)
                self.train_end = fold.train[1]
                self.val_end = fold.validation[1]
                self.final_test_start = fold.final_test[0]
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
        if self.hparams.split_name == "legacy":
            self.test_dataset = RiverWindowDataset(
                self.data,
                self.scaler,
                self.val_end,
                self.data.values.shape[0],
                self.hparams.input_window,
                self.hparams.output_window,
            )
        else:
            self.test_dataset = None

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
        if self.test_dataset is None:
            raise RuntimeError("final test is locked until Session D")
        return self._loader(self.test_dataset, shuffle=False)

    def _loader(self, dataset: RiverWindowDataset, shuffle: bool) -> DataLoader:
        num_workers = int(self.hparams.num_workers)
        configured_persistence = self.hparams.persistent_workers
        persistent_workers = num_workers > 0 and (
            bool(configured_persistence)
            if configured_persistence is not None
            else True
        )
        loader_options = {
            "dataset": dataset,
            "batch_size": self.hparams.batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": river_collate,
            "persistent_workers": persistent_workers,
            "pin_memory": self.hparams.pin_memory,
        }
        if num_workers > 0 and self.hparams.prefetch_factor is not None:
            loader_options["prefetch_factor"] = int(self.hparams.prefetch_factor)
        return DataLoader(
            **loader_options,
        )
