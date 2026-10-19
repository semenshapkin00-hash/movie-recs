"""Recommendations from fixed item models and persistent live user profiles."""

import json
from pathlib import Path

import numpy as np
from scipy import sparse

from movie_recs.fold_in import fold_in
from movie_recs.profiles import UserStore
from movie_recs.ranking import top_k

MODEL_NAMES = ("mf", "neural", "als", "sansa")


class Recommender:
    def __init__(self, artifact_dir: Path, user_db: Path):
        try:
            self.catalog = json.loads((artifact_dir / "catalog.json").read_text(encoding="utf-8"))
            self.report = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
            self.vectors = {}
            for name in ("mf", "neural", "als"):
                with np.load(artifact_dir / f"{name}.npz", allow_pickle=False) as arrays:
                    users, items = arrays["users"], arrays["items"]
                if (
                    users.ndim != 2
                    or items.ndim != 2
                    or users.shape[1] != items.shape[1]
                    or len(users) != len(self.catalog["user_ids"])
                    or len(items) != len(self.catalog["movies"])
                    or not np.isfinite(users).all()
                    or not np.isfinite(items).all()
                ):
                    raise ValueError(f"Incompatible embeddings: {name}")
                self.vectors[name] = (users, items)
            self.sansa = tuple(sparse.load_npz(artifact_dir / f"sansa_w{i}.npz") for i in (1, 2))
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Missing model artifacts in {artifact_dir}. Train all four models first."
            ) from exc
        size = len(self.catalog["movies"])
        if any(w.shape != (size, size) or not np.isfinite(w.data).all() for w in self.sansa):
            raise ValueError("Incompatible SANSA weights")
        self.user_index = {user_id: index for index, user_id in enumerate(self.catalog["user_ids"])}
        self.item_index = {movie["movie_id"]: i for i, movie in enumerate(self.catalog["movies"])}
        self.grams = {name: items.T @ items for name, (_, items) in self.vectors.items()}
        self.default_model = self.report["default_model"]
        self.store = UserStore(user_db, self.catalog["reserved_max_user_id"])

    def user_exists(self, user_id: int) -> bool:
        return user_id in self.user_index or self.store.exists(user_id)

    def validate_movies(self, movie_ids: list[int]) -> None:
        for movie_id in movie_ids:
            if movie_id not in self.item_index:
                raise ValueError(f"Unknown or untrained movie_id: {movie_id}")

    def create_user(self, ratings: dict[int, float]) -> int:
        self.validate_movies(list(ratings))
        return self.store.create(ratings)

    def rate(self, user_id: int, movie_id: int, rating: float) -> None:
        if not self.user_exists(user_id):
            raise KeyError(user_id)
        self.validate_movies([movie_id])
        self.store.rate(user_id, movie_id, rating)

    def movies(self, query: str, limit: int) -> list[dict]:
        query = query.casefold()
        return [movie for movie in self.catalog["movies"] if query in movie["title"].casefold()][
            :limit
        ]

    def recommend(self, user_id: int, k: int, model: str | None = None) -> dict:
        if not self.user_exists(user_id):
            raise KeyError(user_id)
        model = model or self.default_model
        user = self.user_index.get(user_id)
        updates = self.store.ratings(user_id)
        positives = set(self.catalog["positive_items"][user]) if user is not None else set()
        seen = set(self.catalog["seen"][user]) if user is not None else set()
        for movie_id, rating in updates.items():
            item = self.item_index[movie_id]
            seen.add(item)
            if rating >= 4.0:
                positives.add(item)
            else:
                positives.discard(item)
        if not positives:
            scores = np.array(self.catalog["popularity"], dtype=np.float64)
            strategy = "popularity"
        elif model == "sansa":
            indices = sorted(positives)
            profile = sparse.csr_matrix(
                (np.ones(len(indices)), (np.zeros(len(indices), dtype=int), indices)),
                shape=(1, len(self.catalog["movies"])),
            )
            scores = ((profile @ self.sansa[0]) @ self.sansa[1]).toarray().ravel()
            strategy = "history"
        else:
            users, items = self.vectors[model]
            if user is not None and not updates:
                vector = users[user]
                strategy = "trained_user"
            else:
                config = self.report["models"][model].get("fold_in", {})
                vector = fold_in(
                    items,
                    sorted(positives),
                    config.get("regularization", 0.1),
                    config.get("confidence", 20.0),
                    self.grams[model],
                )
                strategy = "als_fold_in" if model == "als" else "profile_fold_in"
            scores = items @ vector
        selected = top_k(scores, seen, k)
        return {
            "user_id": user_id,
            "model": model,
            "strategy": strategy,
            "recommendations": [
                {**self.catalog["movies"][int(item)], "score": float(scores[item])}
                for item in selected
            ],
        }
