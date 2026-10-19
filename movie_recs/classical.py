"""Training adapters for implicit ALS and the official SANSA implementation."""

import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy import sparse

from movie_recs.data import Dataset
from movie_recs.ranking import evaluate


def interaction_matrix(data: Dataset) -> sparse.csr_matrix:
    return sparse.csr_matrix(
        (
            np.ones(len(data.positives), dtype=np.float32),
            (data.positives[:, 0], data.positives[:, 1]),
        ),
        shape=(len(data.user_ids), len(data.movies)),
    )


def train_als(data: Dataset, seed: int, dim: int, output: Path) -> dict:
    from implicit.cpu.als import AlternatingLeastSquares

    started = time.perf_counter()
    config = {"factors": dim, "regularization": 0.1, "alpha": 20.0, "iterations": 30}
    model = AlternatingLeastSquares(
        **config,
        random_state=seed,
        num_threads=2,
        use_cg=False,
    )
    best_ndcg, best_epoch = -1.0, 0
    best_vectors = None
    history = []

    def validate(epoch: int, elapsed: float, loss: float) -> None:
        nonlocal best_ndcg, best_vectors, best_epoch
        metrics = evaluate(
            model.user_factors @ model.item_factors.T, data.train_seen, data.val_targets
        )
        history.append({"epoch": epoch + 1, **metrics})
        if metrics["ndcg@10"] > best_ndcg:
            best_ndcg = metrics["ndcg@10"]
            best_epoch = epoch + 1
            best_vectors = (model.user_factors.copy(), model.item_factors.copy())
        print(f"als    epoch={epoch + 1:02d} val_ndcg@10={metrics['ndcg@10']:.4f}", flush=True)

    model.fit(interaction_matrix(data), show_progress=False, callback=validate)
    users, items = best_vectors
    scores = users @ items.T
    np.savez_compressed(output / "als.npz", users=users, items=items)
    return {
        "implementation": "implicit.cpu.als.AlternatingLeastSquares",
        "library_version": version("implicit"),
        "config": {**config, "seed": seed},
        "fold_in": {"regularization": config["regularization"], "confidence": config["alpha"]},
        "best_epoch": best_epoch,
        "epochs_run": config["iterations"],
        "history": history,
        "training_seconds": round(time.perf_counter() - started, 3),
        "validation": evaluate(scores, data.train_seen, data.val_targets),
        "test": evaluate(scores, data.val_seen, data.test_targets),
    }


def train_sansa(data: Dataset, output: Path) -> dict:
    from sansa import (
        SANSA,
        CHOLMODGramianFactorizerConfig,
        SANSAConfig,
        UMRUnitLowerTriangleInverterConfig,
    )

    started = time.perf_counter()
    config = {"l2": 100.0, "weight_matrix_density": 0.01, "scans": 1, "finetune_steps": 5}
    model = SANSA(
        SANSAConfig(
            l2=config["l2"],
            weight_matrix_density=config["weight_matrix_density"],
            gramian_factorizer_config=CHOLMODGramianFactorizerConfig(),
            lower_triangle_inverter_config=UMRUnitLowerTriangleInverterConfig(
                scans=config["scans"],
                finetune_steps=config["finetune_steps"],
            ),
        )
    )
    interactions = interaction_matrix(data)
    model.fit(interactions)
    w1, w2 = model.weights
    sparse.save_npz(output / "sansa_w1.npz", w1)
    sparse.save_npz(output / "sansa_w2.npz", w2)
    scores = model.forward(interactions).toarray()
    # Diagonal self terms are immaterial: all input films are masked during ranking.
    return {
        "implementation": "sansa==1.1.0 (official glami/sansa)",
        "library_version": version("sansa"),
        "config": {**config, "factorization": "CHOLMOD", "inversion": "UMR"},
        "weights_nnz": [w1.nnz, w2.nnz],
        "training_seconds": round(time.perf_counter() - started, 3),
        "validation": evaluate(scores, data.train_seen, data.val_targets),
        "test": evaluate(scores, data.val_seen, data.test_targets),
    }
