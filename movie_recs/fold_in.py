"""Estimate a user vector with frozen item vectors; no global retraining needed."""

import numpy as np


def fold_in(
    items: np.ndarray,
    positive_items: list[int],
    regularization: float,
    confidence: float,
    gram: np.ndarray | None = None,
) -> np.ndarray:
    """Implicit least squares: observed confidence c, unobserved confidence 1.

    This is the exact ALS user update. For MF/neural exports it is a profile
    approximation in their output space, rather than a new trained ID embedding.
    """
    positive = items[positive_items].astype(np.float64)
    gram = items.T @ items if gram is None else gram
    system = gram.astype(np.float64) + (confidence - 1) * (positive.T @ positive)
    system += regularization * np.eye(items.shape[1])
    target = confidence * positive.sum(axis=0)
    return np.linalg.solve(system, target)
