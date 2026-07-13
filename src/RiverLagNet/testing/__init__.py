"""Utilities for removing transient test outputs without touching test sources."""

from .cleanup import clean_test_artifacts

__all__ = ["clean_test_artifacts"]
