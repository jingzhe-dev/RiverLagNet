import torch

from RiverLagNet.data.datamodule import RiverDataModule


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
