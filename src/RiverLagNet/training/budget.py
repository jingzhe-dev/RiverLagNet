"""Equal-exposure optimizer budget contracts for paired experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingBudget:
    """Resolved effective batch and maximum optimizer-update exposure."""

    effective_batch_size: int
    max_optimizer_steps: int
    gradient_accumulation: int


def resolve_accumulation(physical_batch_size: int, effective_batch_size: int) -> int:
    """Return integral gradient accumulation for an exact effective batch."""
    if physical_batch_size <= 0 or effective_batch_size <= 0:
        raise ValueError("physical and effective batch sizes must be positive")
    if effective_batch_size % physical_batch_size:
        raise ValueError("effective batch size must be divisible by physical batch size")
    return effective_batch_size // physical_batch_size


def resolve_training_budget(
    physical_batch_size: int,
    effective_batch_size: int,
    max_optimizer_steps: int,
) -> TrainingBudget:
    """Resolve one comparable training budget from a physical batch choice."""
    if max_optimizer_steps == 0 or max_optimizer_steps < -1:
        raise ValueError("max optimizer steps must be -1 or positive")
    return TrainingBudget(
        effective_batch_size=effective_batch_size,
        max_optimizer_steps=max_optimizer_steps,
        gradient_accumulation=resolve_accumulation(
            physical_batch_size, effective_batch_size
        ),
    )
