"""The same exact full-catalog retrieval is used in evaluation and the API."""

from collections.abc import Iterable

import numpy as np


def top_k(scores: np.ndarray, seen: Iterable[int], k: int) -> np.ndarray:
    scores = scores.copy()
    scores[list(seen)] = -np.inf
    # A stable sort makes ties deterministic (catalog order is ascending movie ID).
    ranked = np.argsort(-scores, kind="stable")
    return ranked[np.isfinite(scores[ranked])][:k]


def evaluate(
    scores: np.ndarray, seen: list[set[int]], targets: list[set[int]], k: int = 10
) -> dict:
    recalls, ndcgs = [], []
    discounts = 1.0 / np.log2(np.arange(k) + 2)
    for user, relevant in enumerate(targets):
        relevant = relevant - seen[user]
        if not relevant:
            continue
        ranked = top_k(scores[user], seen[user], k)
        hits = np.array([item in relevant for item in ranked], dtype=np.float64)
        recalls.append(float(hits.sum() / len(relevant)))
        ideal = discounts[: min(k, len(relevant))].sum()
        ndcgs.append(float((hits * discounts[: len(hits)]).sum() / ideal))
    if not recalls:
        raise ValueError("No eligible users for evaluation.")
    return {
        f"recall@{k}": float(np.mean(recalls)),
        f"ndcg@{k}": float(np.mean(ndcgs)),
        "evaluated_users": len(recalls),
    }
