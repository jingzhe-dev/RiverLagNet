from __future__ import annotations

import inspect

import pytest
import torch

from RiverLagNet.models.local_multiscale import LocalMultiscaleForecaster


def _inputs(
    *, batch: int = 2, history: int = 31, nodes: int = 3, variables: int = 5
) -> dict[str, torch.Tensor]:
    return {
        "x": torch.randn(batch, history, nodes, variables),
        "x_mask": torch.ones(batch, history, nodes, variables, dtype=torch.bool),
        "x_quality": torch.ones(batch, history, nodes, variables),
        "static": torch.randn(nodes, 2),
        "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "edge_attr": torch.randn(2, 3),
        "time_features": torch.randn(batch, history, 4),
    }


def _model(**overrides: object) -> LocalMultiscaleForecaster:
    kwargs: dict[str, object] = {
        "value_dim": 5,
        "static_dim": 2,
        "time_dim": 4,
        "edge_dim": 3,
        "hidden_dim": 8,
        "output_window": 30,
        "target_dim": 3,
        "scales": (1, 3, 7, 30),
        "num_layers": 2,
        "dropout": 0.0,
    }
    kwargs.update(overrides)
    return LocalMultiscaleForecaster(**kwargs)


def test_local_multiscale_returns_common_shape_and_context_contract() -> None:
    model = _model().eval()
    inputs = _inputs()

    context = model.encode_context(**inputs)

    assert context.history_states.shape == (2, 31, 3, 8)
    assert context.scale_states.shape == (2, 4, 3, 8)
    assert context.horizon_states.shape == (2, 30, 3, 8)
    assert context.prediction.shape == (2, 30, 3, 3)
    assert torch.equal(model(**inputs), context.prediction)


def test_graph_free_model_is_exactly_station_isolated() -> None:
    torch.manual_seed(17)
    model = _model().eval()
    inputs = _inputs()
    baseline = model(**inputs)
    perturbed = {name: value.clone() for name, value in inputs.items()}
    perturbed["x"][:, :, 0] += 100.0
    perturbed["x_quality"][:, :, 0] *= 0.1

    changed = model(**perturbed)

    assert not torch.equal(changed[:, :, 0], baseline[:, :, 0])
    assert torch.equal(changed[:, :, 1:], baseline[:, :, 1:])


def test_future_truth_is_not_a_model_input_and_cannot_change_prediction() -> None:
    torch.manual_seed(23)
    model = _model().eval()
    inputs = _inputs()
    future_truth = torch.randn(2, 30, 3, 3)
    baseline = model(**inputs)
    future_truth.add_(10_000.0)

    repeated = model(**inputs)

    assert "y" not in inspect.signature(model.forward).parameters
    assert "future_truth" not in inspect.signature(model.forward).parameters
    assert torch.equal(repeated, baseline)


def test_fully_missing_nan_values_are_masked_and_finite() -> None:
    model = _model().eval()
    inputs = _inputs()
    inputs["x"].fill_(torch.nan)
    inputs["x_quality"].fill_(torch.nan)
    inputs["x_mask"].fill_(False)

    context = model.encode_context(**inputs)

    assert torch.isfinite(context.history_states).all()
    assert torch.isfinite(context.scale_states).all()
    assert torch.isfinite(context.horizon_states).all()
    assert torch.isfinite(context.prediction).all()


def test_target_and_exogenous_streams_have_disjoint_parameters() -> None:
    encoder = _model().input_encoder
    target_parameters = {parameter.data_ptr() for parameter in encoder.target_encoder.parameters()}
    external_parameters = {parameter.data_ptr() for parameter in encoder.external_encoder.parameters()}

    assert target_parameters
    assert external_parameters
    assert target_parameters.isdisjoint(external_parameters)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_local_multiscale_cuda_bf16_autocast_is_finite() -> None:
    model = _model(hidden_dim=16, scales=(1, 3), num_layers=1).cuda().train()
    inputs = {name: value.cuda() for name, value in _inputs(history=12).items()}

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(**inputs)
        loss = output.square().mean()
    loss.backward()

    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
