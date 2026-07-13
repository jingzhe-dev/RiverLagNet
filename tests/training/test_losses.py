import torch

from RiverLagNet.training.losses import masked_huber_loss


def test_masked_huber_ignores_missing_targets_and_empty_mask_is_differentiable() -> None:
    prediction = torch.tensor([1.0, 100.0], requires_grad=True)
    target = torch.tensor([0.0, 0.0])
    mask = torch.tensor([True, False])
    first = masked_huber_loss(prediction, target, mask)
    changed = masked_huber_loss(torch.tensor([1.0, -999.0]), target, mask)
    assert torch.allclose(first.detach(), changed)
    empty = masked_huber_loss(prediction, target, torch.zeros_like(mask))
    assert empty.item() == 0.0
    empty.backward()
    assert prediction.grad is not None
