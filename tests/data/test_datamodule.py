from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from RiverLagNet.data.datamodule import RiverDataModule


@pytest.mark.parametrize(
    ("split_name", "train_end", "validation_end"),
    [
        ("v02_fold_a", 132, 156),
        ("v02_fold_b", 156, 180),
        ("v02_fold_c", 180, 204),
    ],
)
def test_v02_fold_windows_and_scaler_are_development_only(
    split_name: str, train_end: int, validation_end: int
) -> None:
    module = RiverDataModule(
        num_days=240,
        num_nodes=5,
        input_window=20,
        output_window=10,
        batch_size=4,
        split_name=split_name,
        seed=5,
    )

    module.setup("fit")

    assert module.data is not None and module.scaler is not None
    assert module.train_end == train_end
    assert module.val_end == validation_end
    assert module.final_test_start == 204
    train_targets = set(module.train_dataset.all_target_indices())
    validation_targets = set(module.val_dataset.all_target_indices())
    assert train_targets.isdisjoint(validation_targets)
    assert max(train_targets) < train_end
    assert min(validation_targets) == train_end
    assert max(validation_targets) < validation_end <= module.final_test_start
    expected = []
    for feature in range(module.data.values.shape[-1]):
        values = module.data.values[:train_end, :, feature]
        mask = module.data.observed[:train_end, :, feature]
        expected.append(values[mask].mean())
    assert torch.allclose(module.scaler.mean, torch.stack(expected))
    with pytest.raises(RuntimeError, match="final test is locked until Session D"):
        module.test_dataloader()


def test_datamodule_fits_scaler_on_training_period_and_collates_shared_graph() -> None:
    module = RiverDataModule(
        num_days=180,
        num_nodes=5,
        input_window=20,
        output_window=10,
        batch_size=4,
        seed=5,
    )
    module.setup("fit")
    assert module.data is not None and module.scaler is not None
    expected = []
    for feature in range(module.data.values.shape[-1]):
        values = module.data.values[: module.train_end, :, feature]
        mask = module.data.observed[: module.train_end, :, feature]
        expected.append(values[mask].mean())
    assert torch.allclose(module.scaler.mean, torch.stack(expected))
    batch = next(iter(module.train_dataloader()))
    assert batch["x"].shape == (4, 20, 5, 3)
    assert batch["y"].shape == (4, 10, 5, 3)
    assert batch["static"].shape == (5, 3)
    assert batch["edge_index"].shape[0] == 2
    assert batch["edge_attr"].shape[0] == batch["edge_index"].shape[1]


def test_chronological_splits_have_disjoint_target_timestamps() -> None:
    module = RiverDataModule(num_days=220, input_window=30, output_window=10, seed=9)
    module.setup()
    train_targets = set(module.train_dataset.all_target_indices())
    val_targets = set(module.val_dataset.all_target_indices())
    test_targets = set(module.test_dataset.all_target_indices())
    assert train_targets.isdisjoint(val_targets)
    assert train_targets.isdisjoint(test_targets)
    assert val_targets.isdisjoint(test_targets)


def test_identifiable_datamodule_uses_train_scaler_without_truth_in_batch() -> None:
    module = RiverDataModule(
        scenario="identifiable_v1",
        num_days=180,
        num_nodes=5,
        input_window=20,
        output_window=10,
        batch_size=4,
        seed=5,
    )

    module.setup("fit")

    assert module.synthetic_scenario is not None
    assert module.data is module.synthetic_scenario.data
    assert module.data is not None and module.scaler is not None
    batch = next(iter(module.train_dataloader()))
    assert set(batch).isdisjoint(
        {"true_lag_days", "local_background", "local_events", "routed_load"}
    )
    expected = []
    for feature in range(module.data.values.shape[-1]):
        values = module.data.values[: module.train_end, :, feature]
        mask = module.data.observed[: module.train_end, :, feature]
        expected.append(values[mask].mean())
    assert torch.allclose(module.scaler.mean, torch.stack(expected))


def test_datamodule_rejects_unknown_scenario() -> None:
    with pytest.raises(ValueError, match="scenario"):
        RiverDataModule(scenario="unknown")


def test_dataloader_propagates_worker_and_prefetch_controls() -> None:
    module = RiverDataModule(
        num_days=180,
        num_nodes=5,
        input_window=20,
        output_window=10,
        batch_size=4,
        num_workers=1,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=3,
    )
    module.setup("fit")

    loader = module.train_dataloader()

    assert loader.num_workers == 1
    assert loader.pin_memory is True
    assert loader.persistent_workers is False
    assert loader.prefetch_factor == 3


def test_dataloader_omits_prefetch_when_workers_are_disabled() -> None:
    module = RiverDataModule(
        num_days=180,
        input_window=20,
        output_window=10,
        num_workers=0,
        persistent_workers=True,
        prefetch_factor=4,
    )
    module.setup("fit")

    loader = module.train_dataloader()

    assert loader.num_workers == 0
    assert loader.persistent_workers is False
    assert loader.prefetch_factor is None


def test_identifiable_hydra_config_has_fixed_window_contract() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=["data=synthetic_identifiable_v1"])

    assert cfg.data.scenario == "identifiable_v1"
    assert cfg.data.num_days == 520
    assert cfg.data.input_window == 90
    assert cfg.data.output_window == 30
    assert cfg.model.lag_prior_strength == 8.0
