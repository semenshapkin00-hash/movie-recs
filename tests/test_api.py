import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy import sparse

from movie_recs.api import create_app
from movie_recs.profiles import UserStore


@pytest.fixture
def artifacts(tmp_path):
    catalog = {
        "user_ids": [42],
        "reserved_max_user_id": 610,
        "seen": [[0]],
        "positive_items": [[0]],
        "popularity": [100, 10, 20],
        "movies": [
            {"movie_id": movie, "title": f"Film {movie}", "genres": ["Drama"]}
            for movie in (10, 20, 30)
        ],
    }
    metrics = {
        "default_model": "neural",
        "models": {
            name: {"validation": {}, "test": {}}
            for name in ("popularity", "mf", "neural", "als", "sansa")
        },
    }
    (tmp_path / "catalog.json").write_text(json.dumps(catalog))
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    for name in ("mf", "neural", "als"):
        np.savez(
            tmp_path / f"{name}.npz",
            users=np.array([[1.0, 0.0]]),
            items=np.array([[1.0, 1.0], [0.1, 1.0], [0.9, 0.0]]),
        )
    sparse.save_npz(tmp_path / "sansa_w1.npz", sparse.eye(3, format="csr"))
    sparse.save_npz(
        tmp_path / "sansa_w2.npz", sparse.csr_matrix([[0, 0.1, 0.9], [0.5, 0, 0.3], [0.2, 0.4, 0]])
    )
    return tmp_path


@pytest.fixture
def client(artifacts):
    with TestClient(create_app(artifacts, artifacts / "users.sqlite3")) as test_client:
        yield test_client


@pytest.mark.parametrize("model", ["mf", "neural", "als", "sansa"])
def test_recommendations_use_external_ids_and_exclude_seen(client, model):
    response = client.get(f"/recommendations/42?k=10&model={model}")
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == model
    assert [item["movie_id"] for item in body["recommendations"]] == [30, 20]
    assert body["recommendations"][0]["score"] == pytest.approx(0.9)
    assert client.get("/health").json()["status"] == "ok"
    assert "sansa" in client.get("/models").json()["models"]


@pytest.mark.parametrize(
    "path, status",
    [
        ("/recommendations/99", 404),
        ("/recommendations/0", 422),
        ("/recommendations/42?k=0", 422),
        ("/recommendations/42?k=101", 422),
        ("/recommendations/42?model=invalid", 422),
    ],
)
def test_invalid_requests(client, path, status):
    assert client.get(path).status_code == status


@pytest.mark.parametrize("model", ["mf", "neural", "als", "sansa"])
def test_new_user_and_rating_updates_work_for_each_model(client, model):
    response = client.post("/users", json={})
    assert response.status_code == 201
    user_id = response.json()["user_id"]
    assert user_id > 610
    path = f"/recommendations/{user_id}?model={model}"
    assert client.get(path).json()["strategy"] == "popularity"
    assert client.put(f"/users/{user_id}/ratings/10", json={"rating": 5}).status_code == 200
    result = client.get(path).json()
    assert result["strategy"] != "popularity"
    assert all(row["movie_id"] != 10 for row in result["recommendations"])
    # Replacing a positive rating by a low one removes it from the positive profile.
    client.put(f"/users/{user_id}/ratings/10", json={"rating": 1})
    result = client.get(path).json()
    assert result["strategy"] == "popularity"
    assert all(row["movie_id"] != 10 for row in result["recommendations"])


def test_profiles_survive_app_restart(artifacts):
    database = artifacts / "persist.sqlite3"
    with TestClient(create_app(artifacts, database)) as client:
        user_id = client.post(
            "/users",
            json={
                "ratings": [
                    {"movie_id": 10, "rating": 5},
                    {"movie_id": 20, "rating": 1},
                ]
            },
        ).json()["user_id"]
    with TestClient(create_app(artifacts, database)) as client:
        result = client.get(f"/recommendations/{user_id}?model=sansa").json()
        assert result["strategy"] == "history"
        assert [row["movie_id"] for row in result["recommendations"]] == [30]
        assert client.post("/users", json={}).json()["user_id"] > user_id


@pytest.mark.parametrize("model", ["mf", "neural", "als", "sansa"])
def test_existing_user_ratings_override_the_training_profile(client, model):
    client.put("/users/42/ratings/20", json={"rating": 5})
    result = client.get(f"/recommendations/42?model={model}").json()
    assert [row["movie_id"] for row in result["recommendations"]] == [30]
    assert result["strategy"] != "trained_user"
    client.put("/users/42/ratings/10", json={"rating": 1})
    client.put("/users/42/ratings/20", json={"rating": 1})
    assert client.get(f"/recommendations/42?model={model}").json()["strategy"] == "popularity"


@pytest.mark.parametrize("rating", [0, 6, 3.1, "NaN"])
def test_invalid_ratings_are_rejected(client, rating):
    response = client.post("/users", json={"ratings": [{"movie_id": 10, "rating": rating}]})
    assert response.status_code == 422


def test_unknown_movies_and_duplicate_ratings_do_not_create_a_user(client):
    assert (
        client.post("/users", json={"ratings": [{"movie_id": 999, "rating": 5}]}).status_code == 404
    )
    assert (
        client.post(
            "/users",
            json={"ratings": [{"movie_id": 10, "rating": 5}, {"movie_id": 10, "rating": 1}]},
        ).status_code
        == 422
    )
    assert client.post("/users", json={}).json()["user_id"] == 611
    assert client.put("/users/9999/ratings/10", json={"rating": 5}).status_code == 404
    assert client.get("/recommendations/9999").status_code == 404
    assert client.put("/users/42/ratings/999", json={"rating": 5}).status_code == 404


def test_movie_lookup_matches_titles(client):
    result = client.get("/movies?query=film%2020").json()
    assert [movie["movie_id"] for movie in result["movies"]] == [20]


def test_concurrent_ids_are_unique_and_persisted(tmp_path):
    store = UserStore(tmp_path / "users.sqlite3", 610)
    with ThreadPoolExecutor(max_workers=4) as executor:
        ids = list(executor.map(lambda _: store.create({10: 5.0}), range(12)))
    assert sorted(ids) == list(range(611, 623))
    reopened = UserStore(tmp_path / "users.sqlite3", 610)
    assert reopened.ratings(ids[0]) == {10: 5.0}


def test_missing_artifacts_fail_at_startup(tmp_path):
    with pytest.raises(RuntimeError, match="Missing model artifacts"):
        with TestClient(create_app(tmp_path, tmp_path / "users.sqlite3")):
            pass
