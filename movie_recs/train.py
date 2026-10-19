"""uv run --group train python -m movie_recs.train"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from movie_recs.classical import train_als, train_sansa
from movie_recs.data import Dataset, load_dataset, write_json
from movie_recs.models import MODELS
from movie_recs.ranking import evaluate


def sample_negatives(
    users: np.ndarray, forbidden: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    """Uniform sampling; only training interactions are excluded (no holdout access)."""
    if np.any(forbidden.all(axis=1)[users]):
        raise ValueError("A training user has no unrated candidate movies.")
    negatives = rng.integers(forbidden.shape[1], size=(len(users), count))
    rejected = forbidden[users[:, None], negatives]
    while rejected.any():
        negatives[rejected] = rng.integers(forbidden.shape[1], size=int(rejected.sum()))
        rejected = forbidden[users[:, None], negatives]
    return negatives


@torch.inference_mode()
def embeddings(model: torch.nn.Module, data: Dataset) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    users = model.encode_users(torch.arange(len(data.user_ids))).numpy().copy()
    items = model.encode_items(torch.arange(len(data.movies))).numpy().copy()
    return users, items


def train_model(name: str, data: Dataset, args: argparse.Namespace, output: Path) -> dict:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = MODELS[name](
        len(data.user_ids), len(data.movies), args.dim, torch.from_numpy(data.genres)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    forbidden = np.zeros((len(data.user_ids), len(data.movies)), dtype=bool)
    for user, seen in enumerate(data.train_seen):
        forbidden[user, list(seen)] = True
    best_value, best_epoch, stale = -1.0, 0, 0
    best_weights = None
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        pairs = data.positives[rng.permutation(len(data.positives))]
        negative_items = sample_negatives(pairs[:, 0], forbidden, args.negatives, rng)
        total_loss = 0.0
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start : start + args.batch_size]
            users = torch.from_numpy(batch[:, 0])
            positives = torch.from_numpy(batch[:, 1])
            negatives = torch.from_numpy(negative_items[start : start + args.batch_size])
            user_vectors = model.encode_users(users)
            positive_scores = (user_vectors * model.encode_items(positives)).sum(dim=-1)
            negative_scores = (user_vectors[:, None, :] * model.encode_items(negatives)).sum(dim=-1)
            loss = F.softplus(negative_scores - positive_scores[:, None]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
        user_vectors, item_vectors = embeddings(model, data)
        validation = evaluate(user_vectors @ item_vectors.T, data.train_seen, data.val_targets)
        record = {"epoch": epoch, "bpr_loss": total_loss / len(pairs), **validation}
        history.append(record)
        print(
            f"{name:6s} epoch={epoch:02d} loss={record['bpr_loss']:.4f} "
            f"val_ndcg@10={validation['ndcg@10']:.4f}",
            flush=True,
        )
        if validation["ndcg@10"] > best_value:
            best_value, best_epoch, stale = validation["ndcg@10"], epoch, 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_weights)
    user_vectors, item_vectors = embeddings(model, data)
    scores = user_vectors @ item_vectors.T
    np.savez_compressed(output / f"{name}.npz", users=user_vectors, items=item_vectors)
    torch.save({"model": name, "dim": args.dim, "state_dict": best_weights}, output / f"{name}.pt")
    return {
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "training_seconds": round(time.perf_counter() - started, 3),
        "validation": evaluate(scores, data.train_seen, data.val_targets),
        "test": evaluate(scores, data.val_seen, data.test_targets),
        "history": history,
    }


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=positive_int, default=40)
    parser.add_argument("--patience", type=positive_int, default=6)
    parser.add_argument("--dim", type=positive_int, default=64)
    parser.add_argument("--batch-size", type=positive_int, default=1024)
    parser.add_argument("--negatives", type=positive_int, default=4)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models", nargs="+", choices=[*MODELS, "als", "sansa"], default=[*MODELS, "als", "sansa"]
    )
    args = parser.parse_args()
    if not 0 < args.lr < float("inf"):
        parser.error("--lr must be finite and positive")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    data = load_dataset(args.data_dir)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    counts = np.bincount(data.positives[:, 1], minlength=len(data.movies)).astype(np.float32)
    popularity_scores = np.broadcast_to(counts, (len(data.user_ids), len(counts)))
    report = {
        "dataset": data.info,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "versions": {"numpy": np.__version__, "torch": torch.__version__},
        "protocol": {
            "candidates": "All items with >=1 positive train interaction; no sampled evaluation",
            "validation_mask": "All training ratings, including low ratings",
            "test_mask": "All train and validation ratings, including low ratings",
            "averaging": "Macro-average over warm users with >=1 warm positive target",
            "selection": "Best epoch and default model selected ONLY by validation NDCG@10",
        },
        "models": {
            "popularity": {
                "validation": evaluate(popularity_scores, data.train_seen, data.val_targets),
                "test": evaluate(popularity_scores, data.val_seen, data.test_targets),
            }
        },
    }
    previous = output / "metrics.json"
    if previous.exists():
        old = json.loads(previous.read_text(encoding="utf-8"))
        if old["dataset"]["sha256"] != data.info["sha256"]:
            raise ValueError("Output directory contains metrics for a different dataset.")
        report["models"].update(
            {name: result for name, result in old["models"].items() if name != "popularity"}
        )
    for name in args.models:
        if name in MODELS:
            report["models"][name] = train_model(name, data, args, output)
        elif name == "als":
            report["models"][name] = train_als(data, args.seed, args.dim, output)
        else:
            report["models"][name] = train_sansa(data, output)
    report["default_model"] = max(
        (name for name in report["models"] if name != "popularity"),
        key=lambda name: report["models"][name]["validation"]["ndcg@10"],
    )
    write_json(output / "metrics.json", report)
    write_json(
        output / "catalog.json",
        {
            "user_ids": data.user_ids,
            "movies": data.movies,
            "seen": [sorted(items) for items in data.all_seen],
            "positive_items": [
                sorted(data.positives[data.positives[:, 0] == user, 1].tolist())
                for user in range(len(data.user_ids))
            ],
            "popularity": counts.tolist(),
            "reserved_max_user_id": max(data.user_ids),
        },
    )
    (output / "DATASET_LICENSE.txt").write_text(data.license_text, encoding="utf-8")
    print(f"Default model: {report['default_model']}; artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
