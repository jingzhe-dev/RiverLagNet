import torch

from RiverLagNet.training.losses import masked_huber_loss, masked_nse_loss


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


def test_masked_nse_loss_balances_targets_and_ignores_missing_values() -> None:
    target = torch.tensor([[[[0.0, 0.0]], [[2.0, 4.0]]]])
    prediction = torch.tensor([[[[0.0, 0.0]], [[4.0, 4.0]]]], requires_grad=True)
    mask = torch.ones_like(target, dtype=torch.bool)

    loss = masked_nse_loss(prediction, target, mask)

    assert torch.allclose(loss, torch.tensor(1.0))
    loss.backward()
    assert prediction.grad is not None
    masked = mask.clone()
    masked[..., 0] = False
    assert masked_nse_loss(prediction.detach(), target, masked).item() == 0.0


def test_masked_nse_loss_empty_mask_is_differentiable() -> None:
    prediction = torch.ones(2, 3, requires_grad=True)
    target = torch.zeros_like(prediction)
    loss = masked_nse_loss(prediction, target, torch.zeros_like(prediction, dtype=torch.bool))
    assert loss.item() == 0.0
    loss.backward()
    assert prediction.grad is not None
