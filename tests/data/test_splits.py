import pytest

from RiverLagNet.data.splits import build_v02_folds, select_v02_fold


def test_v02_folds_end_before_locked_final_test() -> None:
    folds = build_v02_folds(1000)

    assert [(fold.train, fold.validation) for fold in folds] == [
        ((0, 550), (550, 650)),
        ((0, 650), (650, 750)),
        ((0, 750), (750, 850)),
    ]
    assert all(fold.validation[1] <= 850 for fold in folds)
    assert all(fold.final_test == (850, 1000) for fold in folds)


def test_v02_folds_are_immutable_and_validate_inputs() -> None:
    fold = select_v02_fold("v02_fold_b", 1000)

    with pytest.raises(AttributeError):
        fold.train = (0, 1)  # type: ignore[misc]
    with pytest.raises(ValueError, match="positive"):
        build_v02_folds(0)
    with pytest.raises(ValueError, match="unknown v0.2 fold"):
        select_v02_fold("legacy", 1000)
