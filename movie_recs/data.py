"""MovieLens download, fixed chronological splits, and training-only vocabularies."""

import csv
import hashlib
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_SHA256 = "696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436"


def download(data_dir: Path) -> Path:
    """Verify the pinned archive; extract only three explicitly named files."""
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "ml-latest-small.zip"
    if not archive.exists():
        with urllib.request.urlopen(DATA_URL, timeout=90) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != DATA_SHA256:
            raise ValueError("MovieLens archive checksum changed; review the dataset version.")
        archive.write_bytes(payload)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError(f"Invalid checksum: remove {archive} and download it again.")
    folder = data_dir / "ml-latest-small"
    folder.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for name in ("movies.csv", "ratings.csv", "README.txt"):
            (folder / name).write_bytes(source.read(f"ml-latest-small/{name}"))
    return folder


def chronological_split(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """80/10/10 by user; movie ID breaks equal-timestamp ties deterministically."""
    by_user: dict[int, list[dict]] = {}
    for row in rows:
        by_user.setdefault(row["user_id"], []).append(row)
    parts: tuple[list[dict], list[dict], list[dict]] = ([], [], [])
    for user_id in sorted(by_user):
        ordered = sorted(by_user[user_id], key=lambda row: (row["timestamp"], row["movie_id"]))
        train_end, val_end = int(0.8 * len(ordered)), int(0.9 * len(ordered))
        for destination, chunk in zip(
            parts, (ordered[:train_end], ordered[train_end:val_end], ordered[val_end:]), strict=True
        ):
            destination.extend(chunk)
    return parts


@dataclass
class Dataset:
    user_ids: list[int]
    movies: list[dict]
    genres: np.ndarray
    positives: np.ndarray
    train_seen: list[set[int]]
    val_seen: list[set[int]]
    all_seen: list[set[int]]
    val_targets: list[set[int]]
    test_targets: list[set[int]]
    info: dict
    license_text: str


def load_dataset(data_dir: Path) -> Dataset:
    folder = download(data_dir)
    rows = []
    with (folder / "ratings.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "user_id": int(row["userId"]),
                    "movie_id": int(row["movieId"]),
                    "rating": float(row["rating"]),
                    "timestamp": int(row["timestamp"]),
                }
            )
    train, validation, test = chronological_split(rows)
    positive_rows = [row for row in train if row["rating"] >= 4.0]
    user_ids = sorted({row["user_id"] for row in positive_rows})
    movie_ids = sorted({row["movie_id"] for row in positive_rows})
    user_index = {user_id: index for index, user_id in enumerate(user_ids)}
    item_index = {movie_id: index for index, movie_id in enumerate(movie_ids)}
    with (folder / "movies.csv").open(encoding="utf-8", newline="") as handle:
        raw_movies = {int(row["movieId"]): row for row in csv.DictReader(handle)}
    movies = [
        {
            "movie_id": movie_id,
            "title": raw_movies[movie_id]["title"],
            "genres": raw_movies[movie_id]["genres"].split("|"),
        }
        for movie_id in movie_ids
    ]
    genre_names = sorted({genre for movie in movies for genre in movie["genres"]})
    genre_index = {genre: index for index, genre in enumerate(genre_names)}
    genres = np.zeros((len(movies), len(genre_names)), dtype=np.float32)
    for index, movie in enumerate(movies):
        for genre in movie["genres"]:
            genres[index, genre_index[genre]] = 1.0

    def sets(interactions: list[dict], positive_only: bool = False) -> list[set[int]]:
        result: list[set[int]] = [set() for _ in user_ids]
        for row in interactions:
            if row["user_id"] in user_index and row["movie_id"] in item_index:
                if not positive_only or row["rating"] >= 4.0:
                    result[user_index[row["user_id"]]].add(item_index[row["movie_id"]])
        return result

    def target_info(interactions: list[dict]) -> dict:
        positive = [row for row in interactions if row["rating"] >= 4.0]
        return {
            "positive_ratings": len(positive),
            "cold_item_positive_ratings": sum(r["movie_id"] not in item_index for r in positive),
            "cold_user_positive_ratings": sum(r["user_id"] not in user_index for r in positive),
        }

    return Dataset(
        user_ids=user_ids,
        movies=movies,
        genres=genres,
        positives=np.array(
            [[user_index[row["user_id"]], item_index[row["movie_id"]]] for row in positive_rows],
            dtype=np.int64,
        ),
        train_seen=sets(train),
        val_seen=sets(train + validation),
        all_seen=sets(rows),
        val_targets=sets(validation, True),
        test_targets=sets(test, True),
        info={
            "name": "MovieLens latest-small (2018-09-26)",
            "url": DATA_URL,
            "sha256": DATA_SHA256,
            "ratings": len(rows),
            "users": len({row["user_id"] for row in rows}),
            "trained_users": len(user_ids),
            "catalog_movies": len(raw_movies),
            "candidate_movies": len(movies),
            "genre_names": genre_names,
            "train_ratings": len(train),
            "validation_ratings": len(validation),
            "test_ratings": len(test),
            "train_positives": len(positive_rows),
            "validation": target_info(validation),
            "test": target_info(test),
            "split": "per-user chronological 80/10/10; ties by movie_id",
            "positive_threshold": 4.0,
        },
        license_text=(folder / "README.txt").read_text(encoding="utf-8"),
    )


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
