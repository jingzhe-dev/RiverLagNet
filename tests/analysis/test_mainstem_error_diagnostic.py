from __future__ import annotations

import torch

from RiverLagNet.analysis.mainstem_error_diagnostic import _fit_scale


def test_fit_scale_returns_zero_for_zero_graph_correction() -> None:
    correction = torch.zeros(4, 3, 2, 3)
    residual = torch.randn_like(correction)
    mask = torch.ones_like(correction, dtype=torch.bool)

    scale = _fit_scale(
        correction,
        residual,
        mask,
        (0,),
        ridge_fraction=0.0,
    )

    assert torch.isfinite(scale).all()
    assert torch.equal(scale, torch.zeros_like(scale))


def test_fit_scale_recovers_target_specific_linear_correction() -> None:
    correction = torch.randn(20, 4, 3, 3)
    true_scale = torch.tensor([0.5, 1.5, 2.0]).view(1, 1, 1, 3)
    residual = correction * true_scale
    mask = torch.ones_like(correction, dtype=torch.bool)

    scale = _fit_scale(
        correction,
        residual,
        mask,
        (0, 1, 2),
        ridge_fraction=0.0,
    )

    assert torch.allclose(scale, true_scale, atol=1e-6)
